from huggingface_hub import hf_hub_download
import pickle

# Download the .pkl file from your Hugging Face dataset repo
file_path = hf_hub_download(repo_id="thaonguyen217/FG-enhanced-SMILES_20M", filename="FG-enhanced-SMILES_200M.pkl")

# Load the dataset from the .pkl file
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

print(f'Number of FG-enhanced SMILES in the dataset: {len(dataset)}')
print(f'Some examples: {dataset[:10]}')
