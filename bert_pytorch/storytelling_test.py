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
from tqdm import tqdm
class BERTInference:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_text(self, starting_text, max_length=128):
        try:
            print("Encoding input text...")
            input_ids = encode(self.tokenizer, starting_text)[0]
            input_ids = [self.vocab["[CLS]"]] + input_ids + [self.vocab["[SEP]"]]
    
            segment_label = [1 for _ in range(len(input_ids))]
    
            print("Creating tensors...")
            generated = torch.tensor(input_ids).unsqueeze(0)
            generated = generated.to(self.device)
            segment_label = torch.tensor(segment_label).unsqueeze(0).to(self.device)
    
            print("Setting model to eval mode...")
            self.model.eval()
    
            with torch.no_grad():
                for _ in tqdm(range(max_length)):
                    # print("Running forward pass...")
                    # print(f"Shape of generated:  {generated.shape}")
                    # print(f"Shape of the segment labels: {segment_label.shape}")
    
                    predictions = self.model.forward(generated, segment_label)
                    predictions = predictions[1].squeeze(0).to(self.device)
                    # print("Predictions: ", predictions)
                    # print("Getting next word...")
                    next_word = torch.argmax(predictions[-1, :], dim=-1).unsqueeze(0)
    
                    if next_word.item() == encode(self.tokenizer, '[SEP]')[0]:
                        print(f"Got stuck here at {next_word.item()} ")
                        break
    
                    # print("Updating generated and segment_label...")
                    # print(f"Found next token: {next_word}")
    
                    # Add the next word to the end of generated
                    generated = torch.cat((generated, next_word.unsqueeze(0)), dim=1)
                    # If the length of generated exceeds max_length, remove the first token
                    if generated.size(1) > max_length:
                        generated = generated[:, 1:]
    
                    #print(decode(self.tokenizer, generated.squeeze().tolist()))
    
            print("Decoding generated text...")
            return decode(self.tokenizer, generated.squeeze().tolist())
        except Exception as e:
            print(f"Error generating text: {e}")
        return None

     
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--epoch", required=True, type=int, help="epoch of the model to load")
    parser.add_argument("-d", "--device", type=str, default='cpu', help="device to run the model on")
    parser.add_argument("-t", "--text", required=True, type=str, help="starting text for the story")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    vocab = tokenizer.get_vocab()
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    model = KinyaStoryBERTTrainer.load_model_from_path(epoch=args.epoch, vocab_size=len(vocab),bert=bert, device=args.device)
    bert_inference = BERTInference(model, tokenizer, device=args.device)
    print(bert_inference.generate_text(args.text))

if __name__ == '__main__':
    main()
