# MoE-Sim-VAE
Mixture-of-Experts Variational Autoencoder for Clustering and Generating from Similarity-Based Representations on Single Cell Data

## Install conda environment
```bash
conda env create -f spec-file.yml
conda activate moesimvae
```

## Run MoE-Sim-VAE on scRNA-sequencing data clustering mouse organs
```bash
cd src
python run_cluster_mouse_organs.py --dir_output PATH/TO/OUTPUT_DIRECTORY --loss_coef_kl_div_code_standard_gaussian 0.2 --loss_coef_clustering 0.8
```

## PyTorch version
A basic PyTorch implementation is provided in the `torch_src` directory. You can run a toy example with:
```bash
cd torch_src
python run_cluster_mouse_organs.py
```
