import argparse
import sys

import torch
sys.path.append('../')
from ..model import BERT
from trainer import KinyaStoryBERTTrainer
from ..dataset import KinyaStoryNewDataset
from transformers import AutoTokenizer
import os
from ..KinyaTokenizer import KinyaTokenizer, encode, decode
from tqdm import tqdm
from  torch.utils.data import DataLoader
import wandb

class KinyaStoryFinetune:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = tokenizer.get_vocab()

    
    def finetune(self, train_dataset, test_dataset, output_path, last_saved_epoch=None, batch_size=64, epochs=10, num_workers=5, with_cuda=True, log_freq=10, corpus_lines=None, cuda_devices=None, on_memory=True, lr=1e-3, adam_weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999):
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None
        trainer = KinyaStoryBERTTrainer(self.model, self.tokenizer, self.device, train_data_loader, test_data_loader, output_path, last_saved_epoch, with_cuda, log_freq, corpus_lines, cuda_devices, on_memory, lr, adam_weight_decay, adam_beta1, adam_beta2,wandb_project_name="project-ablations",wandb_name="kinya-bert-finetuning",wandb_reinit=True)
        
        # Freeze the BERT model for the first 5 epochs
        best_loss = float('inf')
        self.freeze_or_unfreeze_bert(freeze_until_layer=11, freeze=True)
        for epoch in range(epochs):
            print(f"Training epoch {epoch}")

            # Unfreeze the BERT model after 5 epochs
            if epoch == 5:
                self.freeze_or_unfreeze_bert(freeze_until_layer=11, freeze=False)
            trainer.train(epoch=epoch)
            if test_data_loader is not None:
                # Get the average loss for the current epoch
                current_loss = trainer.get_average_loss()
                trainer.test(epoch=epoch)
                # If the current loss is lower than the best loss, save the model and update the best loss
                if current_loss < best_loss:
                    trainer.save(epoch, output_path)
                    best_loss = current_loss
            
    
    def freeze_or_unfreeze_bert(self, freeze_until_layer=11, freeze=False):
        """
        Freeze or unfreeze the BERT model layers ).. Gradually unfreeze the layers during training
        :param freeze_until_layer: Freeze until this layer
        :param freeze: If True, freeze the layers. If False, unfreeze the layers
        """
        layer_num = 0
        for module in self.model.bert.encoder.layer:
            if layer_num < freeze_until_layer:
                for param in module.parameters():
                    param.requires_grad = False if freeze else True
            layer_num += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--epoch", required=True, type=int, help="epoch of the model to load")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="device to run the model on")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")
    parser.add_argument("-c", "--train_dataset", default="kinyastory_data/train_stories.txt", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="kinyastory_data/val_stories.txt", help="test set for evaluate train set")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model_finetuned")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    vocab = tokenizer.get_vocab()
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    model = KinyaStoryBERTTrainer.load_model_from_path(epoch=args.epoch, vocab_size=len(vocab),bert=bert, device=args.device)
    
    kinya_story_finetune = KinyaStoryFinetune(model, tokenizer, device=args.device)
    train_dataset = KinyaStoryNewDataset(args.train_dataset, vocab, seq_len=args.seq_len, on_memory=True)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    kinya_story_finetune.finetune(train_dataset, args.test_dataset, args.output_path, last_saved_epoch=args.epoch, batch_size=64, epochs=50, num_workers=5, with_cuda=True, log_freq=10, corpus_lines=None, cuda_devices=None, on_memory=True, lr=1e-3, adam_weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999)

    print("Finetuning complete")

if __name__ == "__main__":
    main()
