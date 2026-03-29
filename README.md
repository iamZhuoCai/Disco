# Disco

This repository contains the official implementation of the paper “Steering Diffusion Models Towards Credible Content Recommendation” in ICLR 2026.

## Dataset
We use GossipCop, PolitiFact and MHMisinfo datasets in our experiments. The completed datasets after preprocessing can be downloaded from https://huggingface.co/datasets/anony-user-2025/Disco/tree/main

## Requirements
torch==2.3.0

numpy==1.26.4

pandas==2.2.2

## Quick Start
#### PolitiFact
```python Main.py --data=PolitiFact --l2_dacay=0.001 --pref_strength==0.5 --gamma==0.1```
#### GossipCop
```python Main.py --data=GossipCop --l2_dacay=0.001 --pref_strength==1.5 --gamma==0.1```
#### MHMisinfo
```python Main.py --data=MHMisinfo --l2_dacay=0.01 --pref_strength==1 --gamma==0.4```

## Citation
If you find this repository useful, please cite:

```bibtex
@inproceedings{caisteering,
  title={Steering Diffusion Models Towards Credible Content Recommendation},
  author={Cai, Zhuo and Wang, Shoujin and Li, Jin and Zhou, Peilin and Chu, Victor W and Chen, Fang and Zhu, Tianqing and Aggarwal, Charu C},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
