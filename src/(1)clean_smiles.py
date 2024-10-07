import pandas as pd
import argparse
from rdkit.Chem import MolFromSmiles as s2m
from rdkit.Chem import MolToSmiles as m2s

def clean_smiles(csv_data_path, save_smiles_path):
    # Read the input CSV file
    df = pd.read_csv(csv_data_path)
    
    # Extract SMILES and remove duplicates
    clean_smiles_set = {m2s(s2m(sm)) for sm in df['SMILES'].dropna() if s2m(sm) is not None}

    # Create a new DataFrame with cleaned SMILES
    cleaned_df = pd.DataFrame({'SMILES': list(clean_smiles_set)})
    
    # Save the cleaned SMILES to a new CSV file
    cleaned_df.to_csv(save_smiles_path, index=False)

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Clean SMILES from a CSV file.")
    parser.add_argument('--csv_data_path', type=str, required=True, help='Path to the input CSV file containing SMILES.')
    parser.add_argument('--save_smiles_path', type=str, required=True, help='Path to save the cleaned SMILES CSV file.')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Run the cleaning function with the provided arguments
    clean_smiles(args.csv_data_path, args.save_smiles_path)
