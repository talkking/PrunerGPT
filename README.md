# PrunerGPT

This repository contains code to reproduce the key results of the paper [One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models](https://arxiv.org/abs/2310.09499).

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`datautils.py`, `opt.py`, `baichuan.py`) 
* Perform unstructured, n:m and sparse + quantized PrunerGPT compression on OPT and BaiChuan models. (`sparsegpt.py`, `opt.py`, `baichuan.py`)

We note that this SparseGPT implementation is based on our open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0

## Usage

Here are some sample commands to run baselines and sparsification on OPT models, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
See also the CMD-argument documentation.

```
# Run dense baseline
python opt.py facebook/opt-125m c4

# Run magnitude baseline
python opt.py facebook/opt-125m c4 --sparsity .5 --gmp

# Prune to 50\% uniform sparsity with SparseGPT
python opt.py facebook/opt-125m c4 --sparsity .5

# Prune to 50\% mixed sparsity with PrunerGPT
python opt.py facebook/opt-125m c4 --sparsity .5

# Prune to 50\% + 4-bit with PrunerGPT
python opt.py facebook/opt-125m c4 --sparsity .5 --wbits 4
```

To run on other OPT models, replace "facebook/opt-125m" by the HuggingFace name of the corresponding model.
For the 175B model, access must first be requested from Meta and the checkpoint converted to HuggingFace format, then its location can simply be passed as a name to this script.

The BLOOM script `bloom.py` has a very similar interface, however some features are currently only available for OPT, e.g.:

```
# Sparsify BLOOM-176B with PrunerGPT
python bloom.py bigscience/bloom c4 --sparsity .5
```

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with PrunerGPT
python llama.py LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity 0.5
```

In case one would like to save the sparsified model specify path to saved checkpoint via  `--save` flag.

One can optionally log evalution results to W&B with `--log_wandb`. 

## Cite

If you found this work useful, please consider citing:

```
@inproceedings{shao2024one,
  title={One-shot sensitivity-aware mixed sparsity pruning for large language models},
  author={Shao, Hang and Liu, Bei and Qian, Yanmin},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={11296--11300},
  year={2024},
  organization={IEEE}
}
```
