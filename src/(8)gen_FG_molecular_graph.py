import pickle
import torch
import pandas as pd
from rdkit.Chem import MolFromSmiles as s2m
from helpers import get_structure, set_atom_map_num, preprocess_smiles
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

# Define the dimensionality of the atom features
dim = 128

def graph_data(smiles, feature_dict):
    """
    Converts a SMILES string into a FG graph representation (PyTorch Geometric Data object).

    Parameters:
    smiles (str): SMILES string of the molecule.
    feature_dict (dict): Dictionary containing precomputed feature embeddings (from FG knowledge graph).
    label (int/float): Label for the molecule.

    Returns:
    Data: PyTorch Geometric Data object representing the FG graph. Output of this module is input of GCN link prediction model.
    """
    mol = s2m(smiles)
    set_atom_map_num(mol)  # Assign atom mapping numbers
    structure, bonds = get_structure(mol)  # Extract molecular structure and bond information
    
    num_frags = len(structure)  # Number of fragments (nodes)
    num_edges = len(bonds)      # Number of bonds (edges)
    
    # Initialize atom feature matrix with zeros
    atom_features = np.zeros((num_frags, dim))

    # Preprocess the molecular structure to match feature dictionary keys
    new_structure = dict()
    for idx, sm in enumerate(structure):
        new_sm = preprocess_smiles(sm)  # Preprocess SMILES to match the feature dictionary
        new_structure[idx] = {
            'fg_feature': feature_dict.get(new_sm, np.zeros(dim)),  # Get feature or default to zero
            'atom': structure[sm]['atom']  # Get atom list for fragment
        }

    # Process bond connections (edges between fragments)
    new_bonds = []
    for bond in bonds:
        start_idx, end_idx = bond[:2]
        for key, value in new_structure.items():
            if start_idx in value['atom']:
                start_frag = key
            if end_idx in value['atom']:
                end_frag = key
        new_bonds.append([start_frag, end_frag])
    
    # Assign atom features to the feature matrix
    for idx, value in new_structure.items():
        atom_features[idx, :] = value['fg_feature'].detach().numpy()

    # Initialize edge index matrix
    edge_index = torch.zeros((2, num_edges), dtype=torch.long)
    for bond_idx, bond in enumerate(new_bonds):
        start_idx, end_idx = bond
        edge_index[0, bond_idx] = start_idx
        edge_index[1, bond_idx] = end_idx

    # Return the graph as a PyTorch Geometric Data object
    return Data(
        x=atom_features,               # Node features (atom-level)
        edge_index=edge_index,         # Edge list (bond connections)
        batch=torch.zeros(num_frags, dtype=torch.long),  # Batch index (set to 0 for single graphs)
        smiles=smiles                  # SMILES string (optional, for reference)
    )

def main(csv_path, feature_path, save_path):
    # Load the molecular dataset CSV file
    df = pd.read_csv(csv_path)
    SMILES = list(df['SMILES'].values)  # SMILES strings

    # Load precomputed feature embeddings from the pickle file
    with open(feature_path, 'rb') as f:
        feature_dict = pickle.load(f)

    data_list = []
    
    # Convert each SMILES string into a graph
    for i, sm in enumerate(tqdm(SMILES)):
        try:
            g = graph_data(sm, feature_dict)
            data_list.append(g)
        except Exception as e:
            print(f"Error processing {sm}: {e}")
            continue

    # Print the size of the original dataset and the successfully processed data
    print(f"Total molecules: {len(df)}")
    print(f"Processed graphs: {len(data_list)}")
    
    # Save the list of graph data objects to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)

if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Convert SMILES strings to FG graph and save them.")
    
    # Path to the input CSV file (with 'smiles' and 'p_np' columns)
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file storing SMILES.')
    
    # Path to the precomputed feature embeddings (as a pickle file)
    parser.add_argument('feature_path', type=str, help='Path to the KGE functional group embedding file (pickle).')
    
    # Path to save the output processed graph data (as a pickle file)
    parser.add_argument('save_path', type=str, help='Path to save the processed graph structure data (pickle).')
    
    args = parser.parse_args()
    
    # Execute the main function with the provided arguments
    main(args.csv_path, args.feature_path, args.save_path)
