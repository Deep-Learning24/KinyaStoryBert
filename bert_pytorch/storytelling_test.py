import argparse
import sys

import torch
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset import KinyaStoryNewDataset
from transformers import AutoTokenizer
import os
from KinyaTokenizer import KinyaTokenizer, encode, decode
from tqdm import tqdm

class BERTInference:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = tokenizer.get_vocab()

    
    def generate_text(self, starting_text, max_length=128):
        try:
            # Create a template file and insert the starting text
            starting_text_temp_file = 'kinyastory_data/starting_text_temp.txt'
            with open(starting_text_temp_file, 'w') as f:
                f.write(starting_text)
    
            inference_dataset = KinyaStoryNewDataset(corpus_path=starting_text_temp_file, vocab=self.vocab, seq_len=128)
            inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False)
    
            for batch in inference_loader:
                generated = batch['bert_input']
                segment_label = batch['segment_label']
                break
            # Move the generated tensor to the device
            generated = generated.to(self.device)
            segment_label = segment_label.to(self.device)
            # delete the starting text temp file
            os.remove(starting_text_temp_file)
    
            print("Setting model to eval mode...")
            self.model.eval()
    
            with torch.no_grad():
                for _ in tqdm(range(max_length)):
                    predictions = self.model.forward(generated, segment_label)
                    predictions_masked = predictions[0].squeeze(0).to(self.device)
                    predictions_next = predictions[1].squeeze(0).to(self.device)
    
                    # Replace the masked token in the input with the predicted masked token
                    masked_index = (generated == self.vocab["[MASK]"]).nonzero(as_tuple=True)[1]
                    if masked_index.size(0) > 0:
                        next_masked = torch.argmax(predictions_masked[masked_index[0], :], dim=-1).unsqueeze(0)
                        generated[0, masked_index[0]] = next_masked
    
                    # Append the predicted next word token to the input
                    next_word = torch.argmax(predictions_next[-1, :], dim=-1).unsqueeze(0)
    
                    if next_word.item() == encode(self.tokenizer, '[SEP]')[0]:
                        print(f"Got stuck here at {next_word.item()} ")
                        break
    
                    generated = torch.cat((generated, next_word.unsqueeze(0)), dim=1)
    
                    # If the length of generated exceeds max_length, remove the first token
                    if generated.size(1) > max_length:
                        generated = generated[:, 1:]
    
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
