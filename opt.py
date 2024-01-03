import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *
import bisect
import deepspeed
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    from transformers.deepspeed import HfDeepSpeedConfig
    ds_config = "deepspeed.json"
    #ds_config = HfDeepSpeedConfig(ds_config)
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    # config = model.config
    # model = model.train()
    # model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), optimizer=None, config=ds_config)
    # model.config = config
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev, method="pruning", sparsity_way="origin", sensitivity=None, total_weight=None):
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    ###经过embed层之后捕获第一层的输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
        
    layers[0] = layers[0].module
    #layers[0] = layers[0].cpu()
    # model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    # model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    # if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
    #     model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    # if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
    #     model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    # torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')
    if method == "pruning":
        print("Pruning ...")
    else:
        print("Get Sensitivity ...")
        
    if method == "sensitivity":
        if sparsity_way == "layer-level":
            sensitivity = [0]*len(layers)
        elif sparsity_way == "weight-level":
            sensitivity = []*len(layers)
    else:
        sensitivity = sensitivity
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
              continue
            gpts[name] = SparseGPT(subset[name])
            
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        def get_uniform_sparsity(sen):
            def get_step(a, b, s, n):
                d = 2*(s-a*n) / (n*(n-1))
                if d <= (b-a)/(n-1):
                    return d
                else:
                    return 0
            low_bound = .4
            upper_bound = 1.0
            lens = len(layers)
            spa = args.sparsity
            last_layer_num = 0
            sparsity = (lens * spa - last_layer_num) / (lens - last_layer_num)
            if last_layer_num != 0:
                sen = sen[:-last_layer_num]
            num_layer = lens - last_layer_num
            d = get_step(low_bound, upper_bound, sparsity * num_layer, num_layer)
            _, id = torch.sort(sen, dim=-1)
            sen[id] = torch.arange(low_bound, low_bound + num_layer*d, d)[:num_layer]
            sen = torch.cat((sen, torch.ones(last_layer_num)), dim=-1)
            return sen
            
        def get_layer_sparsity(l):
            #sen = torch.as_tensor([ e.item() for e in sensitivity ])
            sen = torch.as_tensor(sensitivity)
            # normalize_sen = sen / sen.sum()
            # sen = normalize_sen * len(layers) * args.sparsity
            # sen = torch.softmax(sen, dim=-1) * len(layers) * args.sparsity
            
            #sen = get_uniform_sparsity(sen)

            
            last_layer_num = 1
            sparsity = (len(layers) * args.sparsity - last_layer_num) / (len(layers) - last_layer_num)
            sen = sen[:-last_layer_num]
            num_layer = len(layers) - last_layer_num
            #import pdb; pdb.set_trace()
            normalize_sen = sen / sen.sum()
            sen = normalize_sen * num_layer * sparsity
            while torch.any(sen>1.0).item():
                sen = torch.softmax(sen, dim=-1) * num_layer * sparsity
            sen = torch.cat((sen, torch.ones(last_layer_num)), dim=-1)
            # import pdb; pdb.set_trace()
            
            # sen = torch.as_tensor([ e.item() for e in sensitivity ])
            # #import pdb; pdb.set_trace()
            # _, top30 = torch.topk(sen, k=int(0.3*sen.size(-1)), largest=True)
            # _, bottom30 = torch.topk(sen, k=int(0.3*sen.size(-1)), largest=False)
            # normalize_sen = torch.full_like(sen, fill_value=0.5)
            # normalize_sen[top30] = 0.6
            # normalize_sen[bottom30] = 0.4
            # sen = normalize_sen
            #import pdb; pdb.set_trace()
            return 1 - sen[l]
        def get_weight_sparsity(layer, name):
            id = bisect.bisect_left(total_weight, sensitivity[layer][name]) 
            lower_bound = 0.3
            upper_bound = 2 * args.sparsity - lower_bound
            sen = lower_bound + id * (upper_bound - lower_bound) / (len(total_weight) - 1 )
            return 1 - sen 
        
        for name in gpts:
            print(i, name)
            if method == "pruning":
                # 稀疏度的方式
                if sparsity_way == "origin":
                    sparsity = args.sparsity
                elif sparsity_way == "layer-level":
                    sparsity = get_layer_sparsity(i)
                elif sparsity_way == "weight-level":
                    sparsity = get_weight_sparsity(i, name)
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )

            elif method == "sensitivity":
                #sensitivity[i] += gpts[name].average_trace()
                from Myhessian import Hessian as hessian
                #dev = "cuda"
                #import os
                #os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
                #model = model.to(dtype=torch.float32)
                #dataloader = [ (e[0].to(dev), e[0].to(dev)) for e in dataloader ]
                # data = (dataloader[0][0].to(dev), dataloader[0][0].to(dev))
                with torch.enable_grad():
                   #model = model.to(dev)
                   #import pdb; pdb.set_trace()
                   dataloader = dataloader[:min(1, len(dataloader))]
                   ds_config = 'deepspeed.json'
                   config = model.config
                   model = model.train()
                   model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), optimizer=None, config=ds_config)
                   model.config = config
                   hes = hessian(model, nn.CrossEntropyLoss(), dataloader=dataloader, cuda=True, local_rank=args.local_rank)
                   hessian_trace = hes.trace(maxIter=50)
                   print(hessian_trace)
                   import pdb; pdb.set_trace()
                if sparsity_way == "layer-level":
                    for name, trace in hessian_trace.items():
                       if name.startswith("model.decoder.layers"):
                           layer = int(name.split('.')[3])
                           sensitivity[layer] += trace
                elif sparsity_way == "weight-level":
                    dict = {}
                    clayer = 0
                    total_weight = []
                    for name, trace in hessian_trace.items():
                       if name.startswith("model.decoder.layers"):
                           layer = int(name.split('.')[3])
                           if clayer < layer:
                               clayer = layer
                               sensitivity.append(dict)
                               dict = {}
                           subname = ".".join(name.split('.')[4:])
                           if subname.endswith(".weight"):
                               dict[subname[:-7]] = trace
                               total_weight.append(trace)
                    sensitivity.append(dict)
                    #print(sensitivity[0])
                    #print(sensitivity, len(sensitivity))
                    total_weight = sorted(total_weight)
                #import pdb; pdb.set_trace()
                
                
            
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if method == "sensitivity":
       if sparsity_way == "weight-level":
           return sensitivity, total_weight
       elif sparsity_way == "layer-level":
           return sensitivity, None
       

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}
 
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()  ### eos remove
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:] ### sos remove
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
        "--sparsity_way", type=str, default="origin", help="Sparsity way"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local_rank"
    )
    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # dataloader = deepspeed.DeepSpeedEngine.dataloader(
    #     dataloader, model, stream=True
    # )
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        #opt_sequential(model, dataloader, DEV)
        sensitivity = opt_sequential(model, dataloader, DEV, method="sensitivity")
        #import pdb; pdb.set_trace()
        #sensitivity = [0.024750037265516694, 0.06347911567546533, 0.04707093544770524, 0.04599794041149252, 0.08218989528002929, 0.044483200854823046, 0.06627523331859564, 0.06360966167135373, 0.06675981033964362, 0.028765330516103305, 0.02846526448869291, 0.04828428689844699, 0.04114862483984094, 0.02663871526765149, 0.021715643293125808, 0.017720799965410095, 0.018353715486242095, 0.02634426509151755, 0.00952890520053451, 0.009690598873342227, 0.009973193956266613, 0.007518922651387205, 0.008155811201672525, 0.06589200420512498]
        #sensitivity = [1513.0305585898777, 1270.8919252737633, 975.4037257357148, 1546.236512639175, 1570.9296292897138, 1710.5916946639418, 2579.875780077182, 2876.4986672811033, 3246.852629712942, 4618.321481762417, 4541.876624815845, 3857.7017805140526, 3339.151764879946, 3211.957446126425, 2282.788240383803, 2428.0685215430767, 1958.729125737127, 1326.0629923152626, 1214.438442518927, 1039.799794204453, 823.1728501649245, 710.086921476227, 738.2685832145203, 1999.661469769194]
        # sensitivity = [2760.0463225179888, 2389.658272931545, 1509.5805233345939, 1674.1605096723863, 1615.3825463528572, 2518.627436470008, 3540.213683296765, 3752.2745131301485, 3431.3886316769367, 3003.973199491281, 4302.309235519924, 5314.452810171587, 3559.5448292352285, 3557.0843941404737, 3054.235714853331, 2583.3156707629414, 1690.0860415907919, 1415.7097451543632, 1237.7923944180052, 1006.734461552773, 760.5471784913988, 768.5393035661929, 687.8921664435711, 1595.6580456374359]
        sensitivity = [16143.3965, 16240.3389, 16212.6426, 16246.7832, 16313.8184, 16333.3408,
        16273.2109, 16285.9521, 16307.0283, 16325.1377, 16356.2842, 16419.8848,
        16469.0078, 16571.5801, 16615.5508, 16777.9473, 16802.4121, 16928.7363,
        17096.1367, 17137.7832, 17233.6055, 17373.3906, 17486.7305, 18078.2461]
        opt_sequential(model, dataloader, DEV, sparsity_way=args.sparsity_way, sensitivity=sensitivity)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)
    
    datalist = ['wikitext2', 'ptb', 'c4']
    #datalist = ['wikitext2']
    for dataset in datalist:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

'''
(model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50272, 2048, padding_idx=1)
        (embed_positions): OPTLearnedPositionalEmbedding(2050, 2048)
        (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (1): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (2): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (3): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (4): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (5): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (6): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (7): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (8): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (9): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (10): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (11): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (12): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (13): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (14): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (15): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (16): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (17): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (18): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (19): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (20): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (21): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (22): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
          (23): OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
              (out_proj): Linear(in_features=2048, out_features=2048, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (lm_head): Linear(in_features=2048, out_features=50272, bias=False)
)
'''
