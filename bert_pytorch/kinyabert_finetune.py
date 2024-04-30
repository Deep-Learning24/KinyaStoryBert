
import argparse
import gc
import sys

import torch
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset import KinyaStoryNewDataset
import os
from KinyaTokenizer import KinyaTokenizer, encode, decode
from tqdm import tqdm
from  torch.utils.data import DataLoader
import wandb

from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="device to run the model on")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-c", "--train_dataset", default="kinyastory_data/train_stories.txt", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="kinyastory_data/val_stories.txt", help="test set for evaluate train set")
    parser.add_argument("-o", "--output_path", default = "output/bert.model_finetuned" , type=str, help="ex)output/bert.model_finetuned")
    parser.add_argument("-p", "--last_saved_epoch", type=int, default=None, help="epoch of last saved model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    

    args = parser.parse_args()

    # Load the pretrained model
    model = AutoModelForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large")
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    vocab = tokenizer.get_vocab()

    # Load your training and validation data
    train_dataset = KinyaStoryNewDataset(args.train_dataset, tokenizer, seq_len=args.seq_len, on_memory=True)
    val_dataset = KinyaStoryNewDataset(args.test_dataset, tokenizer, seq_len=args.seq_len, on_memory=True) if args.test_dataset is not None else None


    # Create a DataLoader for your training and validation data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Define your optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            # Forward pass
            #print(batch)
            outputs= model(**batch)
            print(outputs)
            loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
            print(f"Training loss: {loss}")
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate the model on the validation data
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                val_loss =  loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                # Compute validation metrics here
                print(f"Validation loss: {val_loss}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"{args.output_path}_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()