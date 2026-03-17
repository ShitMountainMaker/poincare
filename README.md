# Generative Recommendation with Semantic IDs (hyper)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


This local repository is named **hyper**. It is based on **GRID** (Generative Recommendation with Semantic IDs), a framework for generative recommendation systems using semantic IDs developed by [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html). This project implements approaches for learning semantic IDs from text embedding and generating recommendations through transformer-based generative models.

## 🚀 Overview

GRID facilitates generative recommendation three overarching steps:

- **Embedding Generation with LLMs**: Converting item text into embeddings using any LLMs available on Huggingface. 
- **Semantic ID Learning**: Converting item embedding into hierarchical semantic IDs using Residual Quantization techniques such as RQ-KMeans, RQ-VAE, RVQ. 
- **Generative Recommendations**: Using transformer architectures to generate recommendation sequences as semantic ID tokens. 


## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/snap-research/GRID.git hyper
cd hyper

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

For the local HPC setup in this repo, use:

```bash
REPO_DIR=/data/user/cwu319/RC/hyper
DATA_ROOT=/data/user/cwu319/RC/hyper/data/amazon_data
DATA_DIR=/data/user/cwu319/RC/hyper/data/amazon_data/beauty
cd "${REPO_DIR}"
source /data/user/cwu319/conda_envs/rec/bin/activate
```

### 1. Data Preparation

Prepare your dataset in the expected format:
```
data/
├── train/       # training sequence of user history 
├── validation/  # validation sequence of user history 
├── test/        # testing sequence of user history 
└── items/       # text of all items in the dataset
```

We provide pre-processed Amazon data explored in the [P5 paper](https://arxiv.org/abs/2203.13366) [4]. The data can be downloaded from this [google drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Embedding Generation with LLMs

Generate embeddings from LLMs, which later will be transformed into semantic IDs. 

```bash
cd /data/user/cwu319/RC/hyper
python -m src.inference experiment=sem_embeds_inference_flat data_dir=/data/user/cwu319/RC/hyper/data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'
```

Output:
`outputs/semantic_embeddings/pickle/merged_predictions_tensor.pt`

### 3. Train and Generate Semantic IDs

If you only want the original semantic ID baseline from the README pipeline, run:

```bash
cd /data/user/cwu319/RC/hyper
sbatch my_job.sh
```

This uses the fixed-path baseline in `my_job.sh`:

```text
RUN_MODE=base_only
RUN_PROXY_METRICS=0
```

Outputs:
`outputs/semantic_id_stage/base`
`outputs/semantic_id_stage/base/checkpoints/last.ckpt`
`outputs/semantic_id_stage/inference/base/pickle/merged_predictions_tensor.pt`

If you want to run all three semantic ID variants:

```bash
cd /data/user/cwu319/RC/hyper
RUN_MODE=all_three sbatch my_job.sh
```

Outputs:
`outputs/semantic_id_stage/base`
`outputs/semantic_id_stage/euc_prefix`
`outputs/semantic_id_stage/hyp_prefix`
`outputs/semantic_id_stage/inference/base`
`outputs/semantic_id_stage/inference/euc_prefix`
`outputs/semantic_id_stage/inference/hyp_prefix`

Proxy metrics are not part of the original README 1-5 pipeline. Run them only after step 3 has produced all three semantic ID outputs:

```bash
cd /data/user/cwu319/RC/hyper
RUN_MODE=analyze_only RUN_PROXY_METRICS=1 sbatch my_job.sh
```

Outputs:
`outputs/semantic_id_stage/proxy_metrics`
`outputs/semantic_id_stage/semantic_id_stage_comparison.csv`


### 4. Train Generative Recommendation Model with Semantic IDs

Train the recommendation model using the learned semantic IDs:

```bash
cd /data/user/cwu319/RC/hyper
python -m src.train experiment=tiger_train_flat \
    data_dir=/data/user/cwu319/RC/hyper/data/amazon_data/beauty \
    semantic_id_path=outputs/semantic_id_stage/inference/base/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4
```

Use `num_hierarchies=4` because the semantic ID inference step appends one additional digit for de-duplication.

Output directory:
`outputs/recommendation_stage/tiger_train`

### 5. Generate Recommendations

Run inference to generate recommendations:

```bash
cd /data/user/cwu319/RC/hyper
python -m src.inference experiment=tiger_inference_flat \
    data_dir=/data/user/cwu319/RC/hyper/data/amazon_data/beauty \
    semantic_id_path=outputs/semantic_id_stage/inference/base/pickle/merged_predictions_tensor.pt \
    ckpt_path=outputs/recommendation_stage/tiger_train/checkpoints/best.ckpt \
    num_hierarchies=4
```

Use the `best.ckpt` checkpoint from step 4. `num_hierarchies=4` matches the de-duplicated semantic ID length.

Output directory:
`outputs/recommendation_stage/tiger_inference`

## Supported Models:

### Semantic ID:

1. Residual K-means proposed in One-Rec [2]
2. Residual Vector Quantization
3. Residual Quantization with Variational Autoencoder [3]

### Generative Recommendation:

1. TIGER [1]

## 📚 Citation

If you use GRID in your research, please cite:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)
- Inspired by recent advances in generative AI and recommendation systems
- Part of this repo is built on top of https://github.com/ashleve/lightning-hydra-template

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact the development team: Clark Mingxuan Ju (mju@snap.com), Liam Collins (lcollins2@snap.com), Bhuvesh Kumar (bhuvesh@snap.com) and Leonardo Neves (lneves@snap.com).

## Bibliography 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.
