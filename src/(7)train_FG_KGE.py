import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import torch.nn.functional as F
import random
import argparse

class KGDataset(Dataset):
    def __init__(self, triples):
        """
        Initialize the dataset with the given triples.

        Parameters:
        - triples (torch.Tensor): Tensor containing the knowledge graph triples.
        """
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Return the triple as tensors
        head, relation, tail = self.triples[idx]
        return torch.tensor(head), torch.tensor(relation), torch.tensor(tail)

class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__()
        
        # Initialize entity and relation embeddings
        self.ent_real = nn.Embedding(num_entities, embedding_dim)
        self.ent_imag = nn.Embedding(num_entities, embedding_dim)
        self.rel_real = nn.Embedding(num_relations, embedding_dim)
        self.rel_imag = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize embeddings using Xavier uniform distribution
        nn.init.xavier_uniform_(self.ent_real.weight)
        nn.init.xavier_uniform_(self.ent_imag.weight)
        nn.init.xavier_uniform_(self.rel_real.weight)
        nn.init.xavier_uniform_(self.rel_imag.weight)

    def forward(self, triples):
        """
        Compute the ComplEx scores for given triples.

        Parameters:
        - triples (torch.Tensor): A tensor containing triples of (head, relation, tail).

        Returns:
        - score (torch.Tensor): Computed scores for the triples.
        """
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        
        # Get embeddings
        h_real = self.ent_real(h)
        h_imag = self.ent_imag(h)
        r_real = self.rel_real(r)
        r_imag = self.rel_imag(r)
        t_real = self.ent_real(t)
        t_imag = self.ent_imag(t)

        # ComplEx score function
        score = torch.sum(h_real * t_real * r_real +
                          h_imag * t_imag * r_real +
                          h_real * t_imag * r_imag -
                          h_imag * t_real * r_imag, dim=1)
        return score

    def loss(self, pos_score, neg_score):
        """
        Compute the loss using sigmoid cross entropy.

        Parameters:
        - pos_score (torch.Tensor): Scores for positive triples.
        - neg_score (torch.Tensor): Scores for negative triples.

        Returns:
        - loss (torch.Tensor): Computed loss.
        """
        return -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score))

def negative_sampling(triples, num_entities):
    """
    Create negative samples by corrupting the head or tail of the triples.

    Parameters:
    - triples (torch.Tensor): Tensor of triples from the knowledge graph.
    - num_entities (int): Total number of entities in the graph.

    Returns:
    - neg_triples (torch.Tensor): Tensor of negative triples.
    """
    neg_triples = triples.clone()
    for i in range(triples.size(0)):
        if random.random() < 0.5:
            neg_triples[i, 0] = random.randint(0, num_entities - 1)  # Replace head
        else:
            neg_triples[i, 2] = random.randint(0, num_entities - 1)  # Replace tail
    return neg_triples

def extract_smiles_embeddings(node_to_idx, model):
    """
    Extracts SMILES embeddings for entities from the given node_to_idx mapping
    and the trained KGE model.

    Args:
        node_to_idx (dict): Mapping from entity names (including SMILES) to indices.
        model (nn.Module): Trained model for retrieving entity embeddings.

    Returns:
        List: List of embeddings for the entities.
        List: Corresponding SMILES strings.
    """
    IDX = []
    SMILES = []

    # Extract indices and SMILES strings from the node_to_idx mapping
    for i in node_to_idx:
        if i.startswith('smiles='):
            IDX.append(node_to_idx[i])
            SMILES.append(i.replace('smiles=', ''))

    em = []  # List to store embeddings

    # Iterate through indices to get embeddings from the model
    for i in trange(len(IDX), desc="Extracting embeddings"):
        # Retrieve the entity embedding and squeeze it to remove unnecessary dimensions
        e = model.get_entity_embedding(torch.tensor([IDX[i]]))[0].squeeze()
        em.append(e)

    return em, SMILES  # Return the embeddings and SMILES


def train(FGKG_path, checkpoint, device, epochs=100, embedding_dim=128, lr=1e-3, batch_size=64):
    """
    Main function to load data, train the ComplEx model, and save checkpoints.

    Parameters:
    - FGKG_path (str): Path to save data of the FG knowledge graph (pkl file).
    - checkpoint (str): Path to save model checkpoints.
    - epochs (int): Number of training epochs.
    - embedding_dim (int): Dimension of the embeddings.
    - lr (float): Learning rate for the optimizer.
    - batch_size (int): Size of batches for training.
    """
    
    # Load data
    with open(FGKG_path, 'rb') as f:
        data = pickle.load(f)
    
    relation_to_idx = data['relation_to_idx']
    node_to_idx = data['node_to_idx']
    triples = data['triples']
    triples = torch.tensor(triples)

    # Initialize dataset and dataloader
    dataset = KGDataset(triples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_nodes = len(node_to_idx)
    num_relations = len(relation_to_idx)

    # Initialize model and optimizer
    model = ComplEx(num_nodes, num_relations, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Positive triples
        for triple in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            triple = torch.hstack([triple[0].unsqueeze(1), triple[1].unsqueeze(1), triple[2].unsqueeze(1)])
            pos_triples = triple.to(device)

            # Generate negative triples
            neg_triples = negative_sampling(pos_triples, num_nodes).to(device)

            # Forward pass
            pos_score = model(pos_triples)
            neg_score = model(neg_triples)

            # Loss calculation
            loss = model.loss(pos_score, neg_score)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader)}")
        torch.save(model.state_dict(), f"{checkpoint}{epoch + 1}.pth")

    em, SMILES = extract_smiles_embeddings(node_to_idx, model)
    feature_dict = dict()
    for i in range(len(SMILES)):
        feature_dict[SMILES[i]] = em[i]
    return feature_dict


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a ComplEx model on a knowledge graph.")
    parser.add_argument('FGKG_path', type=str, help="Path to the FG knowledge graph (pkl file).")
    parser.add_argument('checkpoint', type=str, help="Path to save model checkpoints.")
    parser.add_argument('feature_dict_path', type=str, help="Path to save feature - FG_embedding dictionary (pkl file).")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of the embeddings.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=64, help="Size of batches for training.")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dict = train(args.FGKG_path, args.checkpoint, device, args.epochs, args.embedding_dim, args.lr, args.batch_size)
    with open(args.feature_dict_path, 'wb') as f:
        pickle.dump(feature_dict)
