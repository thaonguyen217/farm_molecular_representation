# FARM for Molecular Representation
Source code for the paper **FARM: Functional Group-Aware Representations for Small Molecules** [paper](https://arxiv.org/pdf/2410.02082) [webpage](https://thaonguyen217.github.io/farm/)
![FARM model](./images/main.jpg)

## Structure of the Repository

The repository is organized into the following directories:

### Directories Description

- **data/**: Contains scripts for downloading datasets.
  - `download_big_corpus.py`: Script to download a large molecular corpus.
  - `download_small_corpus.py`: Script to download a smaller molecular corpus.

- **downstream_tasks/**: Contains scripts for downstream tasks related to molecular representation.
  - `(0)scaffold_split.py`: Script for splitting data based on scaffolds.
  - `(1)classifier.py`: Script for training a classifier on molecular data.
  - `(2)regressor.py`: Script for training a regressor on molecular data.

- **src/**: Contains the source code and utility scripts for processing molecular data and training models.
  - `helpers.py`: Helper functions for various tasks.
  - `(1)clean_smiles.py`: Script for cleaning SMILES strings.
  - `(2)gen_FG_enhanced_SMILES.py`: Script for generating functional group enhanced SMILES.
  - `(3)train_tokenizer.py`: Script for training a tokenizer on molecular data.
  - `(4)train_bert.py`: Script for training a BERT model for molecular representation.
  - `(5)gen_FG_vocab.py`: Script for generating a vocabulary for functional groups.
  - `(6)gen_FG_KG.py`: Script for generating a knowledge graph based on functional groups.
  - `(7)train_FG_KGE.py`: Script for training a functional group knowledge embedding.
  - `(8)gen_FG_molecular_graph.py`: Script for generating molecular graphs based on functional groups.
  - `(9)train_GCN_link_prediction.py`: Script for training a GCN model for link prediction.
  - `(10)gen_contrastive_learning_data.py`: Script for generating data for contrastive learning.
  - `(11)train_contrastive_bert.py`: Script for training a contrastive BERT model.

## Installation

To install the required packages, please run:

```bash
pip install -r requirements.txt
```

## Use FARM to Extract Molecule Embeddings for Target Dataset
### Step 1: Clean Data
To clean the dataset by removing invalid SMILES and converting SMILES to canonical SMILES, run the `(1)clean_smiles.py` script. This script requires two arguments:
- `csv_data_path`: The path to the input CSV file containing SMILES.
- `save_smiles_path`: The path to save the cleaned SMILES CSV file.
**Example**:
```bash
python clean_smiles.py --csv_data_path path/to/input.csv --save_smiles_path path/to/cleaned_smiles.csv
```
*Note: The input CSV file must contain a column named "SMILES".*

### Step 2: Generate FG-Enhanced SMILES
To generate FG-enhanced SMILES, run the `gen_FG_enhanced_SMILES.py` script. This script requires:
- `csv_path`: The path to the input CSV file containing molecule data.
- `save_path`: The path to save the output pickle file.
**Example**:
```bash
python gen_FG_enhanced_SMILES.py --csv_path path/to/cleaned_smiles.csv --save_path path/to/fg_enhanced_smiles.pkl
```

### Step 3: Download Model and Extract Molecular Embeddings
To extract molecular embeddings for FG-enhanced SMILES, you can use the Hugging Face model. Here’s how to do it in Python:
```python
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

# Load the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained('thaonguyen217/farm_molecular_representation')
model = BertForMaskedLM.from_pretrained('thaonguyen217/farm_molecular_representation')

# Example usage
input_text = "N_primary_amine N_secondary_amine c_6-6 1 n_6-6 n_6-6 c_6-6 c_6-6 2 c_6-6 c_6-6 c_6-6 c_6-6 c_6-6 1 2"  # FG-enhanced representation of NNc1nncc2ccccc12
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model(**inputs, output_hidden_states=True)

# Extract atom embeddings from last hidden states
last_hidden_states = outputs.hidden_states[-1][0]  # last_hidden_states: (N, 768) where N is the input length
```
