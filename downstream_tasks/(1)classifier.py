import argparse
import pickle
import numpy as np
import torch
import warnings
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

# Suppress warnings
warnings.filterwarnings("ignore")

# Define command-line arguments
def parse_args():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Train a GRU Classifier for Molecular Representation')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data (pickle file)')
    parser.add_argument('--val_path', type=str, required=True, help='Path to validation data (pickle file)')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data (pickle file)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save the model checkpoint')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--input_size', type=int, default=768, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for the GRU')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in the GRU')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--pos_weight', type=float, default=1, help='Positive class weight for the loss function')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
    parser.add_argument('--sequence_length', type=int, default=200, help='Sequence length for input data')

    return parser.parse_args()

# Set device for GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset for sequence data
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Collate function for padding sequences in a batch
def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    # Pad sequences to have the same length
    X_batch_padded = pad_sequence(X_batch, batch_first=True)
    return X_batch_padded, torch.tensor(y_batch)

# GRU Classifier Model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc0(x)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(out, c0)  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # Get last hidden state
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)  # out: (batch_size, output_size)
        return out

if __name__ == "__main__":
    args = parse_args()
    
    # Load the tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained('thaonguyen217/farm_molecular_representation')
    model = BertForMaskedLM.from_pretrained('thaonguyen217/farm_molecular_representation').to(device)

    # Load training, validation, and test datasets
    with open(args.train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.val_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(args.test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Prepare training data
    SMILES_train = train_data['X']
    Y_train = torch.from_numpy(np.array(train_data['y'], dtype=np.float32))
    
    SMILES_val = val_data['X']
    Y_val = torch.from_numpy(np.array(val_data['y'], dtype=np.float32))
    
    SMILES_test = test_data['X']
    Y_test = torch.from_numpy(np.array(test_data['y'], dtype=np.float32))

    # Generate input features from SMILES strings
    X_train = [model(**tokenizer(sm, return_tensors='pt').to(device), output_hidden_states=True).hidden_states[-1][0] for sm in tqdm(SMILES_train, desc='Prepare training data')]
    X_val = [model(**tokenizer(sm, return_tensors='pt').to(device), output_hidden_states=True).hidden_states[-1][0] for sm in tqdm(SMILES_val, desc='Prepare validation data')]
    X_test = [model(**tokenizer(sm, return_tensors='pt').to(device), output_hidden_states=True).hidden_states[-1][0] for sm in tqdm(SMILES_test, desc='Prepare test data')]

    # Create DataLoader for training and validation
    dataset = SequenceDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = SequenceDataset(X_val, Y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = GRUClassifier(args.input_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    class_weights = torch.tensor([1.0, args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs // 3, eta_min=1e-5)

    best_score = 0
    model.train()
    
    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        all_labels = []
        all_preds = []
        
        # Training Phase
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(outputs.cpu().detach().numpy())
            loss = criterion(outputs, y_batch.squeeze().float())  # Ensure y_batch is float for BCEWithLogitsLoss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Calculate training AUC
        train_auc_score = roc_auc_score(all_labels, all_preds)
        
        # Validation Phase
        model.eval()  # Set model to evaluation mode
        val_labels = []
        val_preds = []
        with torch.no_grad():  # No gradient computation for validation
            for X_val_batch, y_val_batch in val_dataloader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch).squeeze()
                val_labels.extend(y_val_batch.cpu().numpy())
                val_preds.extend(val_outputs.cpu().detach().numpy())
        
        # Calculate validation AUC
        val_auc_score = roc_auc_score(val_labels, val_preds)
        
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {total_loss / len(dataloader):.4f}, '
              f'Train AUC: {train_auc_score:.4f}, Validation AUC: {val_auc_score:.4f}')
        
        # Save model if validation score improves
        if best_score < val_auc_score:
            torch.save(model.state_dict(), args.checkpoint_path)
            best_score = val_auc_score

    # Testing Phase
    test_dataset = SequenceDataset(X_test, Y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRUClassifier(args.input_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()  # Set model to evaluation mode
    
    test_labels = []
    test_preds = []
    with torch.no_grad():  # No gradient computation for validation
        for X_test_batch, y_test_batch in test_dataloader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            test_outputs = model(X_test_batch).squeeze()
            test_labels.extend(y_test_batch.cpu().numpy())
            test_preds.extend(test_outputs.cpu().detach().numpy())

    # Calculate test AUC
    test_auc_score = roc_auc_score(np.array(test_labels), np.array(test_preds))

    print(f'Test AUC: {test_auc_score:.4f}')