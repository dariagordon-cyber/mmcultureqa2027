#!/bin/bash
#SBATCH --job-name=gemma-think-dev
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --gres=gpu:H100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=gemma-think-dev-%j.out

cd ~/mmcultureqa2027

mkdir -p results
export VLLM_WORKER_MULTIPROC_METHOD=spawn

srun \
  --container-image=nvcr.io#nvidia/vllm:26.08-py3 \
  --container-mounts=${PWD}:${PWD} \
  --container-workdir=${PWD} \
  --export=ALL,HF_HOME=${PWD}/hf_cache,PYTHONPATH=${PWD}/container_pkgs \
  python3 batch_inference.py \
    --model gemma \
    --thinking true \
    --input data/dev_en.jsonl \
    --output results/gemma_thinking_dev.jsonl \
