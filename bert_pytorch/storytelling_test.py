import argparse
import sys

import torch
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset import KinyaStoryNewDataset
from transformers import AutoTokenizer
import os
from KinyaTokenizer import  decode
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import torch.nn.functional as F
import math
import wandb
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction

def calculate_nll(predictions, targets):
    criterion = nn.NLLLoss(ignore_index=0)
    loss = criterion(predictions, targets)
    return loss.item()

def calculate_perplexity(loss):
    perplexity = math.exp(loss)
    return perplexity


def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    return score

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    
    scores = rouge.get_scores(candidate, reference)
    return scores


class BERTInference:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = tokenizer.get_vocab()

    wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")
    wandb.init(project="project-ablations", name="kinya-bert-inference", reinit=True)


    
    def generate_text(self, starting_text, max_length=128):
        try:
            if starting_text is None or len(starting_text) == 0:
                print("Starting text is empty")
                return None
            # Create a template file and insert the starting text
            starting_text_temp_file = 'kinyastory_data/starting_text_temp.txt'
            # Create starting_text_temp_file if it does not exist
            if not os.path.exists(starting_text_temp_file):
                os.makedirs(os.path.dirname(starting_text_temp_file), exist_ok=True)
                with open(starting_text_temp_file, 'w') as f:
                    pass

            with open(starting_text_temp_file, 'w') as f:
                f.write(starting_text+'\n')

            inference_dataset = KinyaStoryNewDataset(corpus_path=starting_text_temp_file, vocab=self.vocab, seq_len=128,is_inference=True)
            inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=4, shuffle=False)
            data_iter = tqdm(enumerate(inference_loader),
                                desc=" Inferencing",
                                total=len(inference_loader),
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
            concatinated_text = ""
            
            for i, data in data_iter:
                data = {key: value.to(self.device) for key, value in data.items()}
                generated = data['bert_input']
                segment_label = data['segment_label']
                label = data['bert_label']
                is_next = data['is_next']

                # # Skip this iteration if is_next is empty
                # if is_next.nelement() == 0:
                #     continue
                
            
                # delete the starting text temp file

                #os.remove(starting_text_temp_file)
        
                print("Setting model to eval mode...")
                self.model.eval()
        
                with torch.no_grad():

                    # Observe the generated text

                    print("The Sent text is: ", decode(self.tokenizer, generated.squeeze().tolist()))
                    
                    predictions_next, predictions_masked = self.model.forward(generated, segment_label)
                
            
                    label_loss = calculate_nll(predictions_masked.transpose(1, 2), label)
                    label_perplexity = calculate_perplexity(label_loss)
                    print(f"Label loss: {label_loss}, Label perplexity: {label_perplexity}")
                    print("is_next ",is_next)
                    print("is_next squeeze",is_next.squeeze())

                    next_loss = calculate_nll(predictions_next, is_next.squeeze()) if is_next is not None or is_next.nelement() != 0 else float('inf')
                    next_perplexity = calculate_perplexity(next_loss)
                    print(f"Next loss: {next_loss}, Next perplexity: {next_perplexity}")

                    total_loss = label_loss + next_loss
                    total_perplexity = calculate_perplexity(total_loss)
                    print(f"Total loss: {total_loss}, Total perplexity: {total_perplexity}")
                
                    # predictions_masked =predictions_masked.squeeze(0).to(self.device)
                    # predictions_next = predictions_next.squeeze(0).to(self.device)
                    
                    # Replace the masked tokens in the input with the predicted masked tokens
                    masked_indices = (generated == self.vocab["[MASK]"]).nonzero(as_tuple=True)[1]

                    # Replace the masked tokens in the input with the predicted masked tokens
                    masked_indices = (generated == self.vocab["[MASK]"]).nonzero(as_tuple=True)[1]
                    
                    for i in range(generated.shape[0]):  # iterate over the sequence dimension
                        for idx in masked_indices:
                            next_masked = torch.argmax(predictions_masked[i, idx, :], dim=-1).item()
                            if next_masked is not  self.vocab['[UNK]']:
                                generated[i, idx] = next_masked
                    
                    
                    # Append the predicted next word token to the input
                    next_word = torch.argmax(predictions_next[-1, :], dim=-1).unsqueeze(0)

                    # Repeat next_word along dimension 0 to match the size of generated
                    next_word = next_word.repeat(generated.shape[0], 1)

                    # Early stopping condition
                    if (next_word == self.vocab['[PAD]']).any() or generated.size(1) > max_length:
                        print(f"Stopping generation at padding token {next_word.item()}")
                        return decode(self.tokenizer, generated.squeeze().tolist())
                        


                    generated = torch.cat((generated, next_word), dim=1)

                    # If the length of generated exceeds max_length, remove the first token
                    # if generated.size(1) > max_length:
                    #     generated = generated[:, 1:]
                    
                print("Decoding generated text...")
                decoded_text = decode(self.tokenizer, generated.squeeze().tolist())
                decoded_text = ' '.join(decoded_text)
                #print("Decoded text: ",decoded_text)
                # Claculate the Rouge and Blue
                #print("Calculating BLEU and ROUGE scores...")
                bleu_score = calculate_bleu(starting_text, decoded_text)
                rouge_score = calculate_rouge(starting_text, decoded_text)
                print(f"BLEU score: {bleu_score}, ROUGE score: {rouge_score}")
                wandb_logs = {
                    "Label Loss": label_loss,
                    "Label Perplexity": label_perplexity,
                    "Next Loss": next_loss,
                    "Next Perplexity": next_perplexity,
                    "Total Loss": total_loss,
                    "Total Perplexity": total_perplexity,
                    "BLEU": bleu_score, 
                    "ROUGE": rouge_score,
                    }
                wandb.log(wandb_logs)
                concatinated_text += decoded_text
                print(f"Generated text: {decoded_text}")
            return self.generate_text(concatinated_text, max_length)
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
    model = KinyaStoryBERTTrainer.load_model_from_path(epoch=args.epoch, vocab_size=len(vocab),bert=bert, device=args.device,is_inference=True)
    bert_inference = BERTInference(model, tokenizer, device=args.device)
    print(bert_inference.generate_text(args.text))

if __name__ == '__main__':
    main()
