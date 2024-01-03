from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path

from transformers import LlamaForCausalLM
import torch
import time
model = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/pretrained_model/mrc_上清'
#'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/sparsegpt/exp/ziya/13B/mixgpt30%'
model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
model.seqlen = 2048

# download onnx, compile
zoo_stub = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/sparsegpt/onnx_model/ziya-13B/mixgpt30%.onnx"
batch_size = 1
compiled_model = Engine(model=zoo_stub, batch_size=batch_size)

#dev = "cuda:6"
dev = "cpu"

# run inference (input is raw numpy tensors, output is raw scores)
model = model.to(dev)
n = 10
m = n
sparse = 0
origin = 0
while n >=1:
    inputs = generate_random_inputs(model_to_path(zoo_stub), batch_size)
    t1 = time.time()
    model(torch.as_tensor(inputs[0]))
    t2 = time.time()
    #torch.cuda.empty_cache()
    print("dense model runtime: ", t2 - t1)
    origin += t2 - t1
    t1 = time.time()
    compiled_model(inputs)
    t2 = time.time()
    sparse += t2 - t1
    n = n - 1
    print("sparse model runtime: ", t2 - t1)
    print("speedup ratio: ", origin / sparse)
    del inputs
    

print("dense model runtime: ", origin / m)
print("sparse model runtime: ", sparse / m)
print("speedup ratio: ", origin / sparse)


