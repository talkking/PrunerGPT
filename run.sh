
# run dense baseline
# python llama.py /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-7b-hf c4

# # Run magnitude baseline
# python opt.py facebook/opt-125m c4 --sparsity .5 --gmp

# # Prune to 50\% uniform sparsity with SparseGPT
# python llama.py /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-7b-hf c4 --sparsity .5

# # Prune to full 2:4 sparsity with SparseGPT
# python opt.py facebook/opt-125m c4 --prunen 2 --prunem 4

# # Prune to 50\% + 4-bit with SparseGPT
# python opt.py facebook/opt-125m c4 --sparsity .5 --wbits 4

model=$1
sparsity_way=$2
model_size=$3
sparsity=$4
if [ $# -lt 1 ]; then
  echo "usage: model_name sparsity_way model_size sparsity"
fi
echo $model
save_path=exp/$model/${model_size}B/${sparsity_way}_${sparsity}
if [ $model == "baichuan" ]; then
    if [ $model_size == 7 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/baichuan-7B
    elif [ $model_size == 13 ]; then
        #model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/Baichuan-13B-Base
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/pretrained_model/Waihu-Baichuan2-13B-Chat-ckpt
    fi
    python baichuan.py $model_path c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} --save $save_path
elif [ $model == "llama" ]; then
    if [ $model_size == 7 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-7b-hf
    elif [ $model_size == 13 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/lm-evaluation-harness/llm_models/llama-13b-hf
        #/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-13b-hf
    elif [ $model_size == 30 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-30b-hf-new
    elif [ $model_size == 65 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama-65b-hf-new
    fi
python llama.py $model_path c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} --save $save_path
elif [ $model == "opt" ]; then
    deepspeed --num_gpus 1 --num_nodes 1 opt.py /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/opt-1.3b c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} 
elif [ $model == "llama2" ]; then
    if [ $model_size == 7 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/llama2-7b-hf
    elif [ $model_size == 13 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chencong29/llm/models/Llama-2-13b-hf
    fi
    python llama2.py $model_path c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} --save $save_path
elif [ $model == "bloom" ]; then
    if [ $model_size == 1.7 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/bloom-1b7
    elif [ $model_size == 7 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/llm-fine-tuning/models/public/bloom-7b
    fi
    python bloom.py $model_path c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} --save $save_path
elif [ $model == "ziya" ]; then
    if [ $model_size == 13 ]; then
        model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/shaohang02/pretrained_model/mrc_上清
    fi
    python ziya.py $model_path c4 --sparsity ${sparsity} --sparsity_way ${sparsity_way} --save $save_path
fi  
