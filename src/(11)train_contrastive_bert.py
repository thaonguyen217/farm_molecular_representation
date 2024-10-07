import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BertModel, Trainer, TrainingArguments

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
MAX_LENGTH = 200
MASKED_PERCENTAGE = 0.35
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01

def load_corpus(paths):
    """Load training data from a list of pickle files."""
    train_data_list = []
    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            train_data_list += data
    return train_data_list

def create_data_dict(data_list):
    """Create a dictionary from the list of data samples."""
    return {
        'pos': [sample['pos'] for sample in data_list],
        'neg': [sample['neg'] for sample in data_list],
        'smiles': [sample['smiles'] for sample in data_list]
    }

def encode_smiles(batch, tokenizer):
    """Tokenize SMILES strings and return the encoded inputs."""
    tokenized = tokenizer(batch['smiles'], padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    return {
        'input_ids': tokenized['input_ids'].squeeze(),
        'attention_mask': tokenized['attention_mask'].squeeze(),
        'pos': batch['pos'],
        'neg': batch['neg']
    }

def custom_data_collator(batch, tokenizer):
    """Custom collator to create batches and apply masking."""
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    pos = torch.stack([torch.tensor(item['pos'], dtype=torch.float) for item in batch])
    neg = torch.stack([torch.tensor(item['neg'], dtype=torch.float) for item in batch])

    # Mask tokens in input_ids with a certain probability
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, MASKED_PERCENTAGE)
    probability_matrix[labels == tokenizer.pad_token_id] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()

    input_ids[masked_indices] = tokenizer.mask_token_id
    labels[~masked_indices] = -100  # Ignore non-masked tokens in loss computation

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pos': pos,
        'neg': neg,
    }

def compute_mlm_loss(logits, labels):
    """Compute the masked language modeling loss."""
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels = labels.view(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits, shift_labels)

def contrastive_loss(projected, pos, neg, margin=0.5):
    """Compute contrastive loss."""
    positive_similarity = F.cosine_similarity(projected, pos)
    negative_similarity = F.cosine_similarity(projected, neg)
    return torch.relu(margin - positive_similarity + negative_similarity).mean()

class ContrastiveBERT(nn.Module):
    """Contrastive BERT model with projection head."""
    def __init__(self, pretrained_model, tokenizer, projection_dim=128):
        super(ContrastiveBERT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.projection_head = nn.Linear(self.bert.config.hidden_size, projection_dim)
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.activation = nn.GELU()
        self.projection_ln = nn.LayerNorm(projection_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        mlm_logits = self.mlm_head(self.activation(last_hidden_state))

        # CLS token at position 0 for contrastive loss
        cls_output = last_hidden_state[:, 0]
        projected = self.projection_ln(self.projection_head(cls_output))

        if labels is not None:
            return projected, mlm_logits, labels
        else:
            return projected, mlm_logits

class ContrastiveBERTTrainer(Trainer):
    """Custom Trainer for Contrastive BERT."""
    def __init__(self, train_dataloader=None, eval_dataloader=None, contrastive_margin=0.5, l_mlm=1.0, l_cls=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.contrastive_margin = contrastive_margin
        self.l_mlm = l_mlm
        self.l_cls = l_cls

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the total loss for the model."""
        input_ids = torch.tensor(inputs['input_ids']).to(self.args.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.args.device)
        mlm_label = torch.tensor(inputs['labels']).to(self.args.device)
        pos = torch.tensor(inputs['pos']).to(self.args.device)
        neg = torch.tensor(inputs['neg']).to(self.args.device)

        # Forward pass through the model
        projected, mlm_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_label)

        # Compute losses
        mlm_loss = compute_mlm_loss(mlm_logits, mlm_label)
        contrastive_loss_val = contrastive_loss(projected, pos, neg, margin=self.contrastive_margin)

        # Total loss
        loss = self.l_mlm * mlm_loss + self.l_cls * contrastive_loss_val
        return (loss, mlm_logits) if return_outputs else loss

def main(train_paths, val_path, tokenizer_path, pretrained_model_path, output_dir):
    """Main function to load data, train the model, and save results."""
    # Load training data
    train_data_list = load_corpus(train_paths)

    # Load tokenizer and pretrained model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    model = ContrastiveBERT(pretrained_model_path, tokenizer)

    # Load validation data
    with open(val_path, 'rb') as f:
        val_data_list = pickle.load(f)

    # Create training and validation datasets
    train_data_dict = create_data_dict(train_data_list)
    val_data_dict = create_data_dict(val_data_list)
    train_dataset = Dataset.from_dict(train_data_dict)
    val_dataset = Dataset.from_dict(val_data_dict)

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: encode_smiles(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: encode_smiles(x, tokenizer), batched=True)

    # Create DataLoaders using the custom data collator
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: custom_data_collator(x, tokenizer))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: custom_data_collator(x, tokenizer))

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='../logs',
        logging_steps=10,
        save_steps=30,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize the custom Trainer with dataloaders
    trainer = ContrastiveBERTTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    # Start the training process
    trainer.train()
    trainer.model.bert.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    # Initialize the argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Train a Contrastive BERT model on molecular data.")
    
    # Add arguments for training corpus paths, validation corpus path, tokenizer path,
    # pretrained model path, and output directory
    parser.add_argument(
        '--train_corpus_path',
        type=str,
        nargs='+',
        required=True,
        help='List of paths to training corpus files (pickled).'
    )
    parser.add_argument(
        '--val_corpus_path',
        type=str,
        required=True,
        help='Path to the validation corpus file (pickled).'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to the tokenizer.'
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        required=True,
        help='Path to the pretrained BERT model.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the output model and tokenizer.'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.train_corpus_path, args.val_corpus_path, args.tokenizer_path, args.pretrained_model_path, args.output_dir)
