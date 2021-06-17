# TGSA
TGSA: Protein-Protein Association-Based Twin Graph Neural Networks for Drug Response Prediction with Similarity Augmentation

# Overview
Here we provide an implementation of Twin Graph neural networks with Similarity Augmentation (TGSA) in Pytorch and PyTorch Geometric. The repository is organised as follows:

- `data/` contains the necessary dataset files;
- `models/` contains the implementation of TGDRP and SA;
- `TGDRP_weights` contains the trained weights of TGDRP;
- `utils/` contains the necessary processing subroutines;
- `preprocess_gene.py` preprocessing for genetic profiles;
- `smiles2graph.py` construct molecular graphs based on SMILES;
- `main.py main` function for TGDRP (train or test);

# Requirements
- Pytorch==1.6.0
- PyTorch Geometric==1.6.1

# Implementation
## Step1: Data Preprocessing
`data/CellLines_DepMap/CCLE_580_18281/census_706/` - Raw genetic profiles from CCLE and the processed features. You can also preprocess your own data with `preprocess_gene.py`.

`data/IC50_GDSC/drug_smiles.csv` - SMILES for 170 drugs. You can generate pyg graph object with `smiles2graph.py`

Coming soon!