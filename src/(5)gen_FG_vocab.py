import re
import pickle
import pandas as pd
from rdkit.Chem import MolFromSmiles as s2m
from helpers import (
    get_structure,
    set_atom_map_num,
    preprocess_smiles,
    remove_wildcards,
    get_ring_structure,
)
from tqdm import tqdm
import argparse

def extract_functional_groups(input_csv, output_pkl):
    """
    Extract functional groups from SMILES in a given CSV file to create a FG vocabulary; and save them to a pickle file.

    Parameters:
    - input_csv (str): Path to the input CSV file containing a column 'SMILES'.
    - output_pkl (str): Path to the output pickle file for storing functional groups vocabulary.
    """
    
    # Load the CSV file containing SMILES
    df = pd.read_csv(input_csv)
    SMILES = list(df['SMILES'].values)

    FG_VOCAB = set()  # Set to store unique functional groups

    # Process each SMILES string
    for i, smiles in enumerate(tqdm(SMILES)):
        try:
            # Convert SMILES to RDKit molecule
            mol = s2m(smiles)
            set_atom_map_num(mol)  # Set atom map numbers for the molecule
            structure = get_structure(mol)[0]  # Get the structure dictionary

            # Process each key in the structure
            for sm in structure.keys():
                sm = preprocess_smiles(sm)  # Preprocess the SMILES
                m = s2m(sm)  # Convert processed SMILES to molecule
                m = remove_wildcards(m)  # Remove wildcards from the molecule

                # Check if the SMILES contains ring information
                if bool(re.search(r'\d', sm)):
                    m = get_ring_structure(m)  # Get ring structure if applicable
                    FG_VOCAB.add(sm)  # Add to functional groups vocabulary
                else:
                    FG_VOCAB.add(sm)  # Add to functional groups vocabulary

        except Exception as e:
            # Optionally, log the error or print it for debugging
            pass

        # Save vocabulary every 100,000 iterations
        if i % 100000 == 0:
            with open(output_pkl, 'wb') as f:
                pickle.dump(FG_VOCAB, f)

    # Final save of the vocabulary to the output pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(FG_VOCAB, f)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract functional groups from SMILES in a CSV file.")
    parser.add_argument(
        'input_csv',
        type=str,
        help="Path to the input CSV file containing a column 'SMILES'."
    )
    parser.add_argument(
        'output_pkl',
        type=str,
        help="Path to the output pickle file for storing functional groups vocabulary."
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the extraction function with the provided arguments
    extract_functional_groups(args.input_csv, args.output_pkl)