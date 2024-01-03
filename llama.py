import time

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *
import bisect
from sensitivity.llama13B import  hessian_trace

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    #import pdb;pdb.set_trace()
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, method="pruning", sparsity_way="origin", sensitivity=None, total_weight=None):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    if method == "pruning":
        print("Pruning ...")
    else:
        print("Get Sensitivity ...")
    quantizers = {}
    if method == "sensitivity":
        if sparsity_way == "layer-level":
            sensitivity = [0]*len(layers)
        elif sparsity_way == "weight-level":
            sensitivity = []
            total_weight = []
    else:
        sensitivity = sensitivity
        total_weight = total_weight

    clayer = 0
    sen = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
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
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            ## forward_one_step
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            def get_sensitivity_sparsity(sen):
                ###Method2
                last_layer_num = 2
                sparsity = (len(layers) * args.sparsity - last_layer_num) / (len(layers) - last_layer_num)
                sen = sen[:-last_layer_num]
                num_layer = len(layers) - last_layer_num
                #import pdb; pdb.set_trace()
                normalize_sen = sen / sen.sum()
                sen = normalize_sen * num_layer * sparsity
                while torch.any(sen>1.0).item():
                    sen = torch.softmax(sen, dim=-1) * num_layer * sparsity
                sen = torch.cat((sen, torch.ones(last_layer_num)), dim=-1)
                return sen
                
            def get_uniform_sparsity(sen):
                def get_step(a, b, s, n):
                    d = 2*(s-a*n) / (n*(n-1))
                    if d <= (b-a)/(n-1):
                        return d
                    else:
                        return 0
                low_bound = .3
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
            def get_weight_sparsity(layer, name):
                id = bisect.bisect_left(total_weight, sensitivity[layer][name]) 
                lower_bound = 0.4
                upper_bound = 2 * (1 - args.sparsity) - lower_bound
                sen = lower_bound + id * (upper_bound - lower_bound) / (len(total_weight) - 1 )
                return 1 - sen 
            def get_layer_sparisty(l):
                # if l == 1:
                #     import pdb; pdb.set_trace()
                sen = torch.as_tensor(sensitivity)
                #sen = torch.as_tensor([ e.item() for e in sensitivity ])
                ####Method1
                # normalize_sen = sen / sen.sum()
                # sen = normalize_sen * len(layers) * args.sparsity
                # while torch.any(sen>1.0).item():
                #     sen = torch.softmax(sen, dim=-1) * len(layers) * args.sparsity
                # #import pdb; pdb.set_trace()
                

                ###Method3
                # sen = torch.as_tensor([ e.item() for e in sensitivity ])
                # _, top30 = torch.topk(sen, k=int(0.3*sen.size(-1)), largest=True)
                # _, bottom30 = torch.topk(sen, k=int(0.3*sen.size(-1)), largest=False)
                # normalize_sen = torch.full_like(sen, fill_value=0.5)
                # normalize_sen[top30] = 0.6
                # normalize_sen[bottom30] = 0.4
                # sen = normalize_sen

                sen = get_uniform_sparsity(sen)
                
                return 1 - sen[l]
            
            for name in subset:
                print(i, name)
                if method == "pruning":
                    # 稀疏度的方式
                    if sparsity_way == "origin":
                        sparsity = args.sparsity
                    elif sparsity_way == "layer-level":
                        sparsity = get_layer_sparisty(i)
                    elif sparsity_way == "weight-level":
                        sparsity = get_weight_sparsity(i, name)
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                        sparsity_way=sparsity_way
                    )
                elif method == "sensitivity":
                    # if sparsity_way == "layer-level":
                    #     sensitivity[i] += gpts[name].average_trace().item()
                    # elif sparsity_way == "weight-level":
                    #     if clayer != i:
                    #         sensitivity.append(sen)
                    #         clayer = i
                    #         sen = {}
                    #     else:
                    #         sen[name] = gpts[name].average_trace().item()
                    #     #sensitivity[i][name] = gpts[name].average_trace().items()
                    #     total_weight.append(gpts[name].average_trace().item())
                    from Myhessian import Hessian as hessian
                    dev = "cpu"
                    model = model.to(dtype=torch.float32)
                    dataloader = [ (e[0].to(dev), e[0].to(dev)) for e in dataloader ]
                    # data = (dataloader[0][0].to(dev), dataloader[0][0].to(dev))
                    with torch.enable_grad():
                       model = model.to(dev)
                       #import pdb; pdb.set_trace()
                       dataloader = dataloader[:min(1, len(dataloader))]
                    
                       # ds_config = 'deepspeed.json'
                       # config = model.config
                       # model = model.train()
                       # model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), optimizer=None, config=ds_config)
                       # model.config = config
                    
                       hes = hessian(model, nn.CrossEntropyLoss(), dataloader=dataloader, cuda=False)
                       hessian_trace = hes.trace(maxIter=1)
                       import pdb; pdb.set_trace()
                    if sparsity_way == "layer-level":
                        for name, trace in hessian_trace.items():
                           if name.startswith("model.decoder.layers"):
                               layer = int(name.split('.')[3])
                               sensitivity[layer] += trace
                gpts[name].free()
        if method == "sensitivity" and sparsity_way == "weight-level":
            sensitivity.append(sen)
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if method == "sensitivity":
       if sparsity_way == "weight-level":
           return sensitivity, total_weight
       elif sparsity_way == "layer-level":
           return sensitivity, None
    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        #import pdb; pdb.set_trace()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--sparsity_way", type=str, default="origin", help="Sparsity way"
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        #sensitivity, total_weight = llama_sequential(model, dataloader, DEV, method="sensitivity", sparsity_way=args.sparsity_way)
        #import pdb; pdb.set_trace()
        def get_sensitivity(sparsity_way):
            sensitivity = []
            sen = [0]*len(model.model.layers)
            #import pdb; pdb.set_trace()
            dict = {}
            clayer = 0
            total_weight = []
            for name, trace in hessian_trace.items():
               if name.startswith("model.layers"):
                   layer = int(name.split('.')[2])
                   if clayer < layer:
                       clayer = layer
                       sensitivity.append(dict)
                       dict = {}
                   subname = ".".join(name.split('.')[3:])
                   if subname.endswith(".weight"):
                       dict[subname[:-7]] = trace
                       total_weight.append(trace)
                       sen[layer] += trace
            sensitivity.append(dict)
            total_weight = sorted(total_weight)
            if sparsity_way == "layer-level":
                return sen, None
            elif sparsity_way == "weight-level":
                return sensitivity, total_weight
        if args.sparsity_way == "origin":
            sensitivity, total_weight = None, None
        else:
            sensitivity, total_weight = get_sensitivity(sparsity_way=args.sparsity_way)
        
        llama_sequential(model, dataloader, DEV, sparsity_way=args.sparsity_way, sensitivity=sensitivity, total_weight=total_weight)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)
    testdata = ["wikitext2", "ptb", "c4"]
    for dataset in testdata:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

'''
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
'''