<!-- Drafted by Juntang Wang at 2025-03-01 -->

# GraMixC: Multi-Resolution Graph-Based Clustering for Downstream Prediction

This repository contains the code for the paper "GraMixC: Multi-Resolution Graph-Based Clustering for Downstream Prediction".

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

## Overview

This project implements GraphMixC (GMC) for various datasets including DSMI, MNIST, Boston House, and QM9.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ml4dsmz.git
   cd ml4dsmz
   ```
2. Create and activate a conda environment:

   ```bash
   conda env create -f environment.yml --name ml
   conda activate ml
   ```

## Project Structure

```bash
ml4dsmz/
├── data/                         # Dataset storage
├── documents/overleaf_ml4dsmz_manuscript/
├── notebooks/
│   ├── 000.main_DSMZ-pH.ipynb    # DSMZ-pH dataset implementation
│   ├── 000.main_DSMZ-Temp.ipynb  # DSMZ-Temp dataset implementation
│   ├── 001.main_MNIST.ipynb      # MNIST dataset implementation
│   ├── 002.main_BHouse.ipynb     # BHouse dataset implementation
│   ├── 003.main_CIFAR10.ipynb    # CIFAR10 dataset implementation
│   ├── 004.main_QM9.ipynb        # QM9 dataset implementation
├── out/                          # Output storage
├── scripts/                      # Scripts
├── src/                          # Source code
│   ├── neighbours/
│   ├── models.py
│   ├── utils.py
│   ├── visuals.py
├── environment.yml               # Conda environment
├── main.ipynb                    # EDA notebook
├── README.md                     # This file
├── LICENSE
```

## Usage

The project contains several Jupyter notebooks demonstrating the implementation on different datasets:

0. **DSNI Dataset** (`000.main_DSMZ-pH.ipynb`):
1. **MNIST Dataset** (`001.main_MNIST.ipynb`):
2. **Boston Housing Dataset** (`002.main_BHouse.ipynb`):
3. **CIFAR-10 Dataset** (`003.main_CIFAR10.ipynb`):
4. **QM9 Dataset** (`004.main_QM9.ipynb`):
5. **Synthetic Point Cloud Demo** (`100.demo_syn-pointcloud.ipynb`):
6. **Attention Map Demo** (`101.demo_attn-map.ipynb`):
7. **Tabnets Supplimentary** (`200.demo_tabnet.ipynb`):

## Features

- Graph-based unsupervised learning
- Support for multiple datasets (DSNI, MNIST, Boston House, QM9)
- RNA structure analysis capabilities
- Comprehensive data preprocessing pipelines
- Model evaluation metrics including ARI (Adjusted Rand Index)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite our paper:

[Add citation information here]
