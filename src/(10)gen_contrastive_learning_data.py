import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles as s2m
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn

from helpers import get_structure, set_atom_map_num, preprocess_smiles, get_new_smiles_rep

# Define constants
DIM = 128

class LinkPredictorGNN(nn.Module):
    """Graph Neural Network model for link prediction."""
    
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictorGNN, self).__init__()
        self.gcn = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, data):
        """Forward pass through the model."""
        x, edge_index = data.x, data.edge_index
        x = torch.tensor(x).float()
        x = self.gcn(x, edge_index)
        return x

def sample_integers(N, n):
    """Sample n unique integers from range [0, N)."""
    u = torch.randint(0, N, (1,)).item()
    if n == 1:
        return u
    if n == 2:
        v = torch.randint(0, N, (1,)).item()
        while u == v:
            v = torch.randint(0, N, (1,)).item()
        return u, v

def negative_sampling(data):
    """Perform negative sampling by perturbing nodes."""
    data.x = torch.tensor(data.x)
    pos_sample = data.clone()
    num_nodes = data.x.size(0)

    if num_nodes >= 3:
        # Node perturbation
        u, v = sample_integers(num_nodes, 2)
        data.x[[u, v]] = data.x[[v, u]]  # Swap the features of nodes u and v
        neg_1 = data.clone()

        return pos_sample, neg_1
    else:
        return None

def graph_data(smiles, feature_dict):
    """Generate graph data from SMILES representation."""
    mol = s2m(smiles)
    if mol is not None:
        new_smiles = get_new_smiles_rep(mol)
        if len(new_smiles.split()) > 512:
            new_smiles = ' '.join(new_smiles.split()[:512])

        set_atom_map_num(mol)
        structure, bonds = get_structure(mol)
        num_frags = len(structure)
        atom_features = np.zeros((num_frags, DIM))

        new_structure = {}
        for idx, sm in enumerate(structure):
            new_sm = preprocess_smiles(sm)
            new_structure[idx] = {
                'fg_feature': feature_dict.get(new_sm, np.zeros(DIM)),
                'atom': structure[sm]['atom']
            }

        new_bonds = []
        for bond in bonds:
            start_idx, end_idx = bond[:2]
            start_frag = next(key for key, value in new_structure.items() if start_idx in value['atom'])
            end_frag = next(key for key, value in new_structure.items() if end_idx in value['atom'])
            new_bonds.append([start_frag, end_frag])

        for idx, value in new_structure.items():
            atom_features[idx, :] = value['fg_feature']

        edge_index = torch.zeros((2, len(new_bonds)), dtype=torch.long)
        for bond_idx, bond in enumerate(new_bonds):
            start_idx, end_idx = bond
            edge_index[0, bond_idx] = start_idx
            edge_index[1, bond_idx] = end_idx

        return Data(
            x=atom_features,
            edge_index=edge_index,
            batch=torch.zeros(num_frags, dtype=torch.long),
            smiles=smiles
        ), new_smiles

def main(link_prediction_model_path, save_path, corpus_path, fgkg_embedding_path):
    """Main function to run the link prediction model."""
    # Load the corpus
    df = pd.read_csv(corpus_path)
    SMILES = list(df['SMILES'].values)

    # Load feature dictionary
    with open(fgkg_embedding_path, 'rb') as f:
        feature_dict = pickle.load(f)

    # Initialize the model
    model = LinkPredictorGNN(in_channels=DIM, hidden_channels=DIM)
    model.load_state_dict(torch.load(link_prediction_model_path))

    data_list = []
    for sm in tqdm(SMILES):
        try:
            gd, new_smiles = graph_data(sm, feature_dict)
            pos_sample, neg_sample = negative_sampling(gd)
            pos = model(pos_sample).detach().mean(0).squeeze()
            neg = model(neg_sample).detach().mean(0).squeeze()
            data_list.append({'pos': pos, 'neg': neg, 'smiles': new_smiles})
        except Exception as e:
            print(f"Error processing SMILES {sm}: {e}")

    # Save the results
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Processed {len(data_list)} molecules out of {len(df)}.")

if __name__ == "__main__":
    import argparse

    # Argument parser for input paths
    parser = argparse.ArgumentParser(description='Link Prediction Model for Molecules.')
    parser.add_argument('--link_prediction_model', type=str, required=True,
                        help='Path to the link prediction model checkpoint.')
    parser.add_argument('--fgkg_embedding_path', type=str, required=True,
                        help='Path to the feature dictionary pickle file (obtained from KGE model).')
    parser.add_argument('--corpus_path', type=str, required=True,
                        help='Path to the input corpus CSV file.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save the output pickle file using for training the Contrastive BERT model.')

    args = parser.parse_args()
    main(args.link_prediction_model, args.save_path, args.corpus_path, args.fgkg_embedding_path)