import argparse
import sys

import torch
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset.Kinya_storydataset import KinyaStoryBertDataset
from transformers import AutoTokenizer
import os
from KinyaTokenizer import KinyaTokenizer, encode, decode

class BERTInference:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_text(self, starting_text, max_length=512):
        input_ids = encode(self.tokenizer, starting_text)[0]
        generated = torch.tensor(input_ids).unsqueeze(0)
        generated = generated.to(self.device)

        self.model.eval()

        with torch.no_grad():
            for _ in range(max_length):
                predictions = self.model(generated, None)
                # we will just use the Masked Language Model for prediction
                predictions = predictions[1].squeeze(0).to(self.device)

                # get the predicted next sub-word (in our case, the word)
                next_word = torch.argmax(predictions[-1, :], dim=-1).unsqueeze(0)

                if next_word.item() == self.tokenizer.encode(['[SEP]'])[0]:
                    break

                generated = torch.cat((generated, next_word.unsqueeze(0)), dim=1)

        return decode(generated.squeeze().tolist())
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--epoch", required=True, type=int, help="epoch of the model to load")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="device to run the model on")
    parser.add_argument("-t", "--text", required=True, type=str, help="starting text for the story")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    vocab = tokenizer.get_vocab()
    model = KinyaStoryBERTTrainer.load_model_from_path(epoch=args.epoch, vocab_size=len(vocab), device=args.device)
    bert_inference = BERTInference(model, tokenizer, device=args.device)
    print(bert_inference.generate_text(args.text))
