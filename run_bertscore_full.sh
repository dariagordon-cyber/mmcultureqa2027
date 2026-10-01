#!/bin/bash
#SBATCH --job-name=bertscore-full
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=bertscore-full-%j.out

srun \
  --container-image=nvcr.io#nvidia/vllm:26.08-py3 \
  --container-mounts=${PWD}:${PWD} \
  --container-workdir=${PWD} \
  --export=ALL,PYTHONPATH=${PWD}/container_pkgs \
  python3 eval_bertscore_full.py
