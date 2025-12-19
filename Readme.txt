******************************** To use this implementation, you will need the following Python libraries: ********************************

numpy
pandas
torch (PyTorch ≥ 2.0)
scikit-learn
matplotlib
os
glob
tqdm

(Optional, if embedding extraction is included)
esm
fair-esm

**************************************************************** Guideline to Code Execution: ****************************************************************

This repository implements **Prot-ESM**, an alignment-free framework for predicting **multi-ligand binding residues (MLBRs)** using ESM-2 protein language model embeddings and lightweight convolutional neural networks.

--------------------------------------------------
1. Dataset Preparation
--------------------------------------------------

The raw dataset consists of **per-protein residue labels** and **precomputed feature matrices**.

Required directories:
- `features_final/`  → per-protein feature files (NumPy `.npy`)
- `labels/`          → per-protein residue-level labels (NumPy `.npy`)

Each protein must have:
- A feature file of shape **L × 1316**
- A label file of shape **L**

Feature composition per residue:
- ESM-2 embeddings: 1280 dimensions
- Physicochemical descriptors: 10 dimensions
- Structural descriptors: 6 dimensions
- Evolutionary descriptors (stub): 20 dimensions

--------------------------------------------------
2. Independent Test-Set Training and Evaluation
--------------------------------------------------

After verifying the dataset structure, the first model evaluation is performed on an **independent test set**.

The training, validation, and test split is handled inside the training script.

Execute: python train_eval_prot_esm.py

This script:
- Loads feature and label pairs
- Trains the Prot-ESM CNN model
- Evaluates performance on the held-out test set
- Reports metrics including:
  - AUROC
  - AUPR
  - F1-score
  - Accuracy
  - Sensitivity
  - Sensitivity at equal positives (Sens_eqpos)
  - Sensitivity at FPR = 5% (Sens@FPR5)

--------------------------------------------------
3. 10-Fold Cross-Validation
--------------------------------------------------

To assess model generalizability, **10-fold cross-validation** is performed at the protein level.

Execute: python cv_10fold_prot_esm.py

This script:
- Automatically detects the number of proteins
- Splits proteins into 10 folds
- Trains and evaluates the model on each fold
- Aggregates metrics across folds
- Reports **mean ± standard deviation** for:
  - AUROC
  - AUPR
  - F1-score
  - Accuracy

--------------------------------------------------
4. ROC Curve Generation
--------------------------------------------------

ROC curves can be generated for:
- Independent test-set evaluation
- 10-fold cross-validation (per-fold or averaged)

ROC plotting utilities are provided in a Jupyter notebook: mlbr.ipynb

The notebook:
- Loads saved prediction scores and ground-truth labels
- Generates ROC curves using `sklearn.metrics`
- Plots and saves publication-quality figures

--------------------------------------------------
5. Model Architecture
--------------------------------------------------

Prot-ESM employs:
- Per-residue feature matrices (L × 1316)
- Lightweight 1D convolutional layers
- Binary classification head for MLBR prediction

The architecture is designed to:
- Avoid multiple sequence alignments (MSAs)
- Reduce preprocessing time
- Maintain competitive performance with MSA-based methods (e.g., MERIT)

--------------------------------------------------
6. Output
--------------------------------------------------

The code produces:
- Evaluation tables for feature ablation studies
- Independent test-set results
- 10-fold cross-validation metrics
- ROC curve figures for manuscript inclusion

--------------------------------------------------
7. Notes
--------------------------------------------------

- The implementation is **fully alignment-free**
- Suitable for low-homology and metagenomic proteins
- Scales efficiently to large protein datasets
- Designed for reproducible, residue-level MLBR prediction

******************************************************* End of README *******************************************************


