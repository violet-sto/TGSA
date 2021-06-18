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

## Requirements
- Pytorch==1.6.0
- PyTorch Geometric==1.6.1

# Implementation
## Step1: Data Preprocessing
`data/CellLines_DepMap/CCLE_580_18281/census_706/` - Raw genetic profiles from CCLE and the processed features. You can also preprocess your own data with `preprocess_gene.py`.

`data/similarity_augment/` - Directory `edge` contains edges of heterogeneous graphs; directory `parameter` contains parameters needed in finetuning TGDRP or TGDRP_pre; directory `dict` contains dictionaries for mapping between drug data or cell line data. 

`data/Drugs/drug_smiles.csv` - SMILES for 170 drugs. You can generate pyg graph object with `smiles2graph.py`

`data/PANCANCER_IC_82833_580_170.csv` - There are 82833 ln(IC50) values across 580 cel lines and 170 drugs.

## Step2: Model Training/Testing
You can run `python main.py -mode train` to train TGDRP or run `python main.py -mode test` to test trained TGDRP.

## Step3: Similarity Augment
First, you can run `similarity_augment.py` to generate edges of heterogeneous graphs and necessary parameters for SA.

Then, you can run `main_SA.py` to fine-tune TGDRP/TGDRP_pre.
`python main.py -mode train -pretrain 0/1` to fine-tune TGDRP/TGDRP_pre.
`python main.py -mode test -pretrain 0/1` to test fine-tuned SA/SA_pre.  

# License
MIT