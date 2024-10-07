import pandas as pd
from tqdm import trange
from rdkit.Chem import MolFromSmiles as s2m
from helpers import get_new_smiles_rep
import pickle
import argparse

# Function to process the molecular data from a CSV file and save the processed SMILES strings to a pickle file.
def process_data(csv_path, save_path):
    """
    Processes the molecular data from a CSV file, generates FG-enhanced SMILES representations,
    and saves them in a pickle file.

    Parameters:
    csv_path (str): Path to the CSV file containing molecular data with at least the 'SMILES' column.
    save_path (str): Path to save the processed data as a pickle file.

    The input CSV file should have:
      - 'SMILES' column: A column containing SMILES representations of molecules.
      - Optional 'label' column: A column containing labels (if applicable).

    The function limits the length of the processed SMILES to 512 tokens.
    """
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Reset index and remove the old index column
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    CORPUS = []  # List to store the processed SMILES representations
    # LABEL = []  # Uncomment this if you want to store labels alongside SMILES

    # Iterate through the DataFrame
    for i in trange(len(df)):
        sm = df.loc[i, 'SMILES']  # Retrieve the SMILES string
        mol = s2m(sm)  # Convert SMILES string to a molecule object using RDKit
        if mol is not None:
            # Generate new SMILES representation
            new_smiles = get_new_smiles_rep(mol)
            
            # Limit the length of the new SMILES representation to 512 tokens
            if len(new_smiles.split()) > 512:
                new_smiles = ' '.join(new_smiles.split()[:512])
            
            # Append the processed SMILES to the corpus
            CORPUS.append(new_smiles.strip())
            
            # Uncomment if you want to append labels to a separate list
            # LABEL.append(df.loc[i, 'label'])
        
        # Save intermediate results every 5 million iterations to avoid memory overflow
        if i % 5000000 == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(CORPUS, f)
    
    # Save the final processed SMILES data as pickle file for training BERT model or txt file for training tokenizer
    # Optionally, you can save labels by using a dictionary: data = {'X': CORPUS, 'y': LABEL}
    with open(save_path, 'wb') as f:
        pickle.dump(CORPUS, f)

    txt = '\n'.join(CORPUS)
    with open(save_path.replace('pkl', 'txt'), 'w') as f:
        f.write(txt)

if __name__ == '__main__':
    # Argument parser for running the script from the command line
    parser = argparse.ArgumentParser(description='Process molecular data and save it as a pickle file.')
    
    # Argument for the input CSV file path
    parser.add_argument('csv_path', type=str, help="Path to the input CSV file containing molecule data.")
    
    # Argument for the output pickle file path
    parser.add_argument('save_path', type=str, help="Path to save the output pickle file.")

    args = parser.parse_args()
    
    # Call the data processing function with the parsed arguments
    process_data(args.csv_path, args.save_path)
