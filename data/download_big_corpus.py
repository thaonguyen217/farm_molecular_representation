from datasets import load_dataset

# Load the dataset from the Hugging Face repository
dataset = load_dataset("thaonguyen217/FG-enhanced-SMILES_200M")

# Explore the first few examples (if the dataset is structured)
print(f'Number of FG-enhanced SMILES in the dataset: {len(dataset)}')
print(f'Some examples: {dataset[:10]}')
