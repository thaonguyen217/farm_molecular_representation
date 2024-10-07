import numpy as np
import torch
import pickle
import math
import warnings
from tqdm import tqdm
from datasets import Dataset
from transformers import (PreTrainedTokenizerFast, BertForMaskedLM, 
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

# Ignore warnings
warnings.filterwarnings("ignore")

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(train_corpus_path, val_corpus_path, tokenizer_path, pretrained_model, output_dir, max_length, masked_percentage):
    """
    Main function to train a BERT model for masked language modeling.

    Args:
        train_corpus_path (str): Path to the training corpus.
        val_corpus_path (str): Path to the validation corpus.
        tokenizer_path (str): Path to the pretrained tokenizer.
        pretrained_model (str): Path to the pretrained BERT model.
        output_dir (str): Directory to save model outputs.
        max_length (int): Maximum sequence length for tokenization.
        masked_percentage (float): Percentage of tokens to mask during training.
    """

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def encode_sentence(batch):
        """Encodes the sentences in the batch using the tokenizer."""
        return tokenizer(batch['text'], padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

    # Load corpus and create a dataset
    with open(train_corpus_path, 'rb') as f:
        train_corpus = pickle.load(f)
    with open(val_corpus_path, 'rb') as f:
        val_corpus = pickle.load(f)

    # Create datasets
    train_dataset = Dataset.from_dict({'text': train_corpus})
    train_dataset = train_dataset.map(encode_sentence, batched=True)
    val_dataset = Dataset.from_dict({'text': val_corpus})
    val_dataset = val_dataset.map(encode_sentence, batched=True)

    # Define the data collator for masked language modeling (MLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=masked_percentage
    )

    # Load the pretrained BERT model
    model = BertForMaskedLM.from_pretrained(pretrained_model).to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir='../logs',
        logging_steps=10,
        save_steps=10000,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train BERT for Masked Language Modeling')
    parser.add_argument('--train_corpus_path', type=str, required=True, help='Path to the training corpus')
    parser.add_argument('--val_corpus_path', type=str, required=True, help='Path to the validation corpus')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased', help='Path to the pretrained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model outputs')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum length of input sequences')
    parser.add_argument('--masked_percentage', type=float, default=0.35, help='Percentage of tokens to mask')
    
    args = parser.parse_args()
    
    # Call the main function with command-line arguments
    main(args.train_corpus_path, args.val_corpus_path, args.tokenizer_path, args.pretrained_model, args.output_dir, args.max_length, args.masked_percentage)
