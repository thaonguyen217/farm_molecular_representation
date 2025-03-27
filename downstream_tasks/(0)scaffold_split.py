import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
import argparse
from sklearn.utils import shuffle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def read_data(input_csv_path, random_seed=98):
    """Read the CSV file and extract SMILES and labels."""
    df = pd.read_csv(input_csv_path)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print("Columns in the DataFrame:", df.columns)
    SMILES = []
    LABELS = []

    for i in range(len(df)):
        smiles = df.loc[i, 'SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            label = df.loc[i, 'label']  # You can modify this to extract more labels if needed
            LABELS.append(label)
            SMILES.append(smiles)

    return SMILES, LABELS

def scaffold_split(smiles_list, labels, train_frac=0.8, valid_frac=0.1, test_frac=0.1, random_seed=98):
    """Perform a scaffold split on the dataset."""
    # Check that the number of SMILES and labels are the same
    assert len(smiles_list) == len(labels), "The number of SMILES strings and labels must be the same."

    # Use DeepChem's scaffold splitter
    splitter = dc.splits.ScaffoldSplitter()

    # Create a NumpyDataset with SMILES as the 'ids'
    dataset = dc.data.NumpyDataset(X=np.zeros(len(smiles_list)), y=np.array(labels), ids=smiles_list)

    # Perform the scaffold split on the dataset
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset, 
        frac_train=train_frac, 
        frac_valid=valid_frac, 
        frac_test=test_frac,
        seed=random_seed
    )

    # Retrieve the SMILES strings and labels for each split
    train_smiles, train_labels = train_dataset.ids, train_dataset.y
    valid_smiles, valid_labels = valid_dataset.ids, valid_dataset.y
    test_smiles, test_labels = test_dataset.ids, test_dataset.y

    return (train_smiles, train_labels), (valid_smiles, valid_labels), (test_smiles, test_labels)

def save_data(smiles, labels, output_csv_path):
    """Save SMILES and labels to a CSV file."""
    df = pd.DataFrame({'smiles': smiles, 'label': labels})
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Process molecule data.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--train_output', type=str, required=True, help='Path to save the training data CSV file.')
    parser.add_argument('--val_output', type=str, required=True, help='Path to save the validation data CSV file.')
    parser.add_argument('--test_output', type=str, required=True, help='Path to save the test data CSV file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Read data
    SMILES, LABELS = read_data(args.input_csv, random_seed=args.seed)

    # Shuffle the SMILES and labels
    SMILES, LABELS = shuffle(SMILES, LABELS, random_state=args.seed)

    # Perform the scaffold split
    (train_smiles, train_labels), (val_smiles, val_labels), (test_smiles, test_labels) = scaffold_split(SMILES, LABELS)

    # Save the results to CSV files
    save_data(train_smiles, train_labels, args.train_output)
    save_data(val_smiles, val_labels, args.val_output)
    save_data(test_smiles, test_labels, args.test_output)
