import argparse
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
import torch
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="GRU Regressor for Molecular Representation")
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to save the model checkpoint')
    parser.add_argument('--input_size', type=int, default=768, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for the GRU model')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for the model')
    
    return parser.parse_args()

# Load and prepare data
def load_data(train_path, val_path, test_path):
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    X_batch_padded = pad_sequence(X_batch, batch_first=True)
    return X_batch_padded, torch.tensor(y_batch)

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=1):
        super(GRURegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc0(x)
        out = self.dropout(out)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(out, c0)
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained('thaonguyen217/farm_molecular_representation')
    model = BertForMaskedLM.from_pretrained('thaonguyen217/farm_molecular_representation').to(device)

    # Load data
    train_data, val_data, test_data = load_data(args.train_path, args.val_path, args.test_path)

    # Prepare training data
    SMILES_train = train_data['X']
    Y_train = np.array(train_data['y'], dtype=np.float32)
    Y_train = torch.from_numpy(Y_train)

    SMILES_val = val_data['X']
    Y_val = np.array(val_data['y'], dtype=np.float32)
    Y_val = torch.from_numpy(Y_val)

    SMILES_test = test_data['X']
    Y_test = np.array(test_data['y'], dtype=np.float32)
    Y_test = torch.from_numpy(Y_test)

    # Normalization
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    Y_train = (Y_train - y_mean) / y_std
    Y_val = (Y_val - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std

    # Load inputs
    X_train = []
    for sm in tqdm(SMILES_train, desc='Loading training data'):
        inputs = tokenizer(sm, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]
            X_train.append(last_hidden_states)

    X_val = []
    for sm in tqdm(SMILES_val, desc='Loading validation data'):
        inputs = tokenizer(sm, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]
            X_val.append(last_hidden_states)

    X_test = []
    for sm in tqdm(SMILES_test, desc='Loading test data'):
        inputs = tokenizer(sm, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]
            X_test.append(last_hidden_states)

    # Create datasets and loaders
    train_dataset = SequenceDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataset = SequenceDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    model = GRURegressor(args.input_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs // 3, eta_min=1e-5)

    best_score = float('inf')
    model.train()
    for epoch in range(args.num_epochs):
        train_loss, val_loss = 0, 0

        # Training Phase
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()

            train_outputs = outputs * y_std + y_mean
            y_train_batch = y_batch * y_std + y_mean
            loss = criterion(train_outputs, y_train_batch.float())
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_rmse_score = np.sqrt(train_loss / len(train_loader))

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch).squeeze()

                val_outputs = val_outputs * y_std + y_mean
                y_val_batch = y_val_batch * y_std + y_mean

                loss = criterion(val_outputs, y_val_batch.float())
                val_loss += loss.item()

        val_rmse_score = np.sqrt(val_loss / len(val_loader))

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Train rmse: {train_rmse_score:.4f}, Validation rmse: {val_rmse_score:.4f}')

        if best_score > val_rmse_score:
            torch.save(model.state_dict(), args.checkpoint)
            print('Saved model to state dict!')
            best_score = val_rmse_score

        model.train()  # Set model back to training mode

    # TEST
    test_dataset = SequenceDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = GRURegressor(args.input_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    test_predictions = []
    with torch.no_grad():
        for X_test_batch, _ in test_loader:
            X_test_batch = X_test_batch.to(device)
            test_outputs = model(X_test_batch).squeeze()
            test_predictions.append(test_outputs.cpu().numpy())  # Collect predictions as NumPy arrays

    test_predictions = np.array(test_predictions).flatten()

    y_std = y_std.cpu().numpy()
    y_mean = y_mean.cpu().numpy()
    test_predictions = test_predictions * y_std + y_mean

    y_test = (Y_test * y_std + y_mean).cpu().numpy()

    # Calculate and print test metrics
    print('esol')
    print("Test RMSE:", mean_squared_error(y_test, test_predictions, squared=False))
    print("Test MAE:", mean_absolute_error(y_test, test_predictions))
    print("Test R2:", r2_score(y_test, test_predictions))