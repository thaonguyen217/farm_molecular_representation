import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import DataLoader, Dataset
import pickle
from tqdm import tqdm
import numpy as np

class GraphDataset(Dataset):
    """
    Custom dataset for graph data, inheriting from PyTorch Geometric Dataset.
    """
    def __init__(self, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        self.transform = transform
        self.pre_transform = pre_transform
        super(GraphDataset, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if self.transform is not None:
            data = self.transform(data)
        return data


class LinkPredictorGNN(nn.Module):
    """
    Graph Neural Network for link prediction.
    """
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictorGNN, self).__init__()

        # Define GCN layer
        self.gcn = torch_geometric.nn.GCNConv(in_channels, hidden_channels)

        # Define MLPs for edge prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Output size is 1 for BCEWithLogitsLoss
        )

    def forward(self, data):
        """
        Forward pass through the GNN.
        """
        x, edge_index, ptr = data.x, data.edge_index, data.ptr
        x = torch.tensor(x).float()

        # Extract node embeddings using GCN
        x = self.gcn(x, edge_index)

        # Prepare all possible pairs of node indices for edge prediction within the batch
        edge_index_pairs = torch.empty((0, 2), dtype=torch.long, device=x.device)
        for j, i in enumerate(ptr[1:]):
            combinations = torch.combinations(torch.arange(i, device=x.device), r=2)
            num_elements = combinations.size(0)
            num_to_select = int(0.6 * num_elements)
            indices = torch.randperm(num_elements, device=combinations.device)[:num_to_select]
            selected_combinations = combinations[indices]
            edge_index_pairs = torch.vstack([edge_index_pairs, selected_combinations])
            current_edge_index = edge_index[:, ptr[j - 1]:ptr[j]]
            edge_index_pairs = torch.vstack([edge_index_pairs, current_edge_index.T])
            indices = torch.randperm(edge_index_pairs.size(0))
            edge_index_pairs = edge_index_pairs[indices]

        # Extract embeddings for each node pair
        node_i = edge_index_pairs[:, 0]
        node_j = edge_index_pairs[:, 1]

        x_i = x[node_i]
        x_j = x[node_j]

        # Concatenate node embeddings to form edge embeddings
        edge_embeddings = torch.cat([x_i, x_j], dim=-1)

        # Predict edge existence
        edge_logits = self.edge_mlp(edge_embeddings)

        return edge_logits, edge_index_pairs, x


def train_model(data_path, checkpoint_path, atom_feature_dim=128, hidden_channels=128, batch_size=2, lr=1e-4, epochs=5, model_name='LINK_PREDICTION1'):
    """
    Train the link prediction model.

    Args:
        data_path (str): Path to the dataset file.
        checkpoint_path (str): Path to save model checkpoints.
        atom_feature_dim (int): Feature dimension for atoms.
        hidden_channels (int): Number of hidden channels for GNN.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train.
        model_name (str): Name of the model for saving checkpoints.
    """
    # Load training data
    with open(data_path, 'rb') as f:
        train_data_list = pickle.load(f)

    # Initialize model, optimizer, and loss function
    model = LinkPredictorGNN(in_channels=atom_feature_dim, hidden_channels=hidden_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Prepare DataLoader
    train_dataloader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)

    # Train the model
    model.train()
    for e in range(epochs):
        train_loss = 0
        for c, data in enumerate(tqdm(train_dataloader, desc=f'Epoch {e + 1}')):
            optimizer.zero_grad()
            data.x = np.vstack([data.x[i] for i in range(len(data.x))])
            edge_logits, edge_index_pairs, _ = model(data)
            edge_index = data.edge_index

            # Process edge index tensor --> source index < target index
            for i in range(edge_index.size(1)):
                if edge_index[0, i] > edge_index[1, i]:
                    edge_index[0, i], edge_index[1, i] = edge_index[1, i], edge_index[0, i]

            # Create labels
            edge_labels = torch.zeros(len(edge_index_pairs))
            for i, pair in enumerate(edge_index_pairs):
                for p in data.edge_index:
                    if torch.equal(p, pair):
                        edge_labels[i] = 1

            loss = criterion(edge_logits.squeeze(), edge_labels.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(f'Training loss: {train_loss / len(train_dataloader)}')
        # Save checkpoint every 500,000 steps
        if c % 500000 == 0:
            torch.save(model.state_dict(), f'{checkpoint_path}{model_name}_epoch{e + 1}.pth')


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train Link Prediction Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file (pickle file).')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save model checkpoints.')
    parser.add_argument('--atom_feature_dim', type=int, default=128, help='Feature dimension for atoms.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels for GNN.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--model_name', type=str, default='link_prediction_model', help='Name of the model for saving checkpoints.')

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        atom_feature_dim=args.atom_feature_dim,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        model_name=args.model_name
    )