import torch
from transformers import LlamaForCausalLM

model = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/sparsegpt/exp/ziya/13B/ziya_mixgpt70%'
model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
model.seqlen = 2048

w1 = model.model.embed_tokens.weight.shape[0]
w2 = model.model.embed_tokens.weight.shape[1]

x = torch.randint(w1, (1, w2))

onnx_model_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/sparsegpt/onnx_model/ziya-13B/mixgpt70%/mixgpt70%.onnx'

torch.onnx.export(model, x, onnx_model_path, verbose=True)

print("Sucessfully convert pytorch to onnx.")


