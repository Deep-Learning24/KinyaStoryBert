
import argparse
import gc
import math
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
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
def collate_fn(batch):
    # Filter out None items and items with None elements
    batch = [item for item in batch if item is not None and all(element is not None for element in item)]

    # Pad the sequences to the maximum sequence length
    input_ids = [item["input_ids"].squeeze() for item in batch]
    attention_mask = [item["attention_mask"].squeeze() for item in batch]
    token_type_ids = [item["token_type_ids"].squeeze() for item in batch]
    # labels should be exactly the same as input_ids
    labels = [item["input_ids"].squeeze() for item in batch]

    # Pad the input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    # Ensure token_type_ids has the same shape as input_ids
    if token_type_ids.shape != input_ids.shape:
        token_type_ids = torch.zeros_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

def calculate_perplexity(loss):
    clipped_loss = np.clip(loss, None, 50)  # clip to avoid overflow
    perplexity = np.exp(clipped_loss)
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
def load_model(model, model_path,device='cpu'):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.to(device)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model path {model_path} does not exist")
        model.to(device)
        return model

def generate_text(model, tokenizer, input_text, max_len=128,device='cpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    encoded_input = tokenizer(input_text, return_tensors='pt',truncation=True, padding='max_length', max_length=max_len)
    encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
    output = model(**encoded_input)
    logits = output.logits
    predicted_index = torch.argmax(logits, dim=-1)
    predicted_text = tokenizer.decode(predicted_index[0])
    return predicted_text

def custom_loss_fn(output, target, ignore_indices=[tokenizer.pad_token_id, -100]):
    loss = nn.CrossEntropyLoss(reduction='none')(output, target)
    for idx in ignore_indices:
        loss[target == idx] = 0
    return loss.mean()


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
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load the pretrained model
    model = AutoModelForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large")
    model.to(args.device)
    # Print the model architecture
    print(model)

    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    vocab = tokenizer.get_vocab()
    # Load the model from the last saved epoch
    if args.last_saved_epoch is not None:
        model = load_model(model, f"{args.output_path}_epoch_{args.last_saved_epoch}.pth", args.device)

    # Load your training and validation data
    train_dataset = KinyaStoryNewDataset(args.train_dataset, tokenizer, seq_len=args.seq_len, on_memory=True)
    val_dataset = KinyaStoryNewDataset(args.test_dataset, tokenizer, seq_len=args.seq_len, on_memory=False) if args.test_dataset is not None else None


    # Create a DataLoader for your training and validation data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset is not None else None

    # Define your optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = custom_loss_fn
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler()
     
    # Train the model
    total_train_loss = 0
    total_val_loss = 0
    total_train_perplexity = 0
    total_val_perplexity = 0
    total_bleu = 0
    #total_rouge = 0
    best_bleu = 0
    best_loss = np.inf
    
    config = {  
        "epochs": args.epochs,
        "device": args.device,
        "hidden": args.hidden,
        "layers": args.layers,
        "attn_heads": args.attn_heads,
        "seq_len": args.seq_len,
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "output_path": args.output_path,
        "last_saved_epoch": args.last_saved_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr
    }


    # # Iterate over each batch in the training DataLoader
    # print("Loading training data...")
    # try:
    #     for i, batch in enumerate(tqdm(train_loader)):
    #         print(f"Loaded batch {i}")
    # except Exception as e:
    #     print(f"Error loading batch {i}: {e}")

    # # Iterate over each batch in the validation DataLoader
    # # Iterate over each batch in the validation DataLoader
    # if val_loader is not None:
    #     print("Loading validation data...")
    #     try:
    #         for i, batch in enumerate(tqdm(val_loader)):
    #             print(f"Loaded batch {i}")
    #     except Exception as e:
    #         print(f"Error loading batch {i}: {e}")
    
    # print("Done loading data")
    # return

    wandb.login(key="3644f3d76a394594794c1b136a20f75303e871ba")
    wandb.init(
        project="project-ablations", 
        config=config,
        reinit=True,
        name="KinyaBERT-large-finetuning",
        notes="Fine-tuning KinyaBERT-large on KinyaStoryNewDataset",
        tags=["kinyabert", "kinyastory", "finetuning"],
        id="kinyabert-large-finetun"
        )
    wandb.watch(model)
    for epoch in range(args.epochs):
        # Clean the GPU cache
        torch.cuda.empty_cache()
        gc.collect()
    
        model.train()
        train_loss = 0
        val_loss = 0
    
        with tqdm(train_loader, desc="Epoch {}".format(epoch)) as progress_bar:
            for batch in progress_bar:
                # Forward pass
                # Get the input and labels from the batch
                #print(batch)
                inputs = {key: tensor.to(args.device) for key, tensor in batch.items()}
                # Assuming `inputs` is a dictionary containing the input tensors
                # Before this line: outputs = model(**inputs)
                inputs_shape = inputs['input_ids'].shape
                #print(inputs_shape)
                if len(inputs_shape) != 2:
                    # Reshape or pad your inputs here
                    # This is just an example, you need to adjust this according to your needs
                    inputs['input_ids'] = inputs['input_ids'].reshape(-1, 2)

                
                input_ids = inputs['input_ids']
                
                labels = batch['input_ids'].to(args.device)
                # print(inputs)
                # print(inputs['input_ids'].shape)
                # print(inputs['token_type_ids'].shape)
                with autocast():
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    train_loss = loss.item()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()), 'perplexity': '{:.3f}'.format(calculate_perplexity(loss.item()) )})
                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        total_train_loss += train_loss
        perplexity = calculate_perplexity(train_loss)
        total_train_perplexity += perplexity
        candidate = " ".join(decode(tokenizer, inputs["input_ids"].squeeze().tolist()))
        #print(f"Predicted text: {candidate}")
        reference = " ".join(decode(tokenizer, labels.squeeze().tolist()))
        #print(f"Reference text: {reference}")
    
        bleu_score = calculate_bleu(reference, candidate)
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{args.output_path}_epoch_{epoch}.pth")
            gc.collect()
            print(f"Model saved at epoch {epoch}")
    
        wandb.log({"training_loss": train_loss, "train perplexity": perplexity, "bleu_score": bleu_score})
    
        # # Evaluate the model on the validation data
        # model.eval()
        # with torch.no_grad():
        #     with tqdm(val_loader, desc="Epoch {}".format(epoch)) as progress_bar:
        #         for batch in progress_bar:
        #             # move the tensors to the device
        #             inputs = {key: tensor.squeeze(0).to(args.device) for key, tensor in batch.items() if key != "labels"}
        #             labels = batch["input_ids"].to(args.device)
        #             outputs = model(**inputs)
        #             loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        #             val_loss += loss.item()
        #             progress_bar.set_postfix({'validation_loss': '{:.3f}'.format(loss.item()), 'perplexity': '{:.3f}'.format(calculate_perplexity(loss.item()))})
    
        # total_val_loss += val_loss
        # val_perplexity = calculate_perplexity(val_loss)
    
        
    
        # total_val_perplexity += val_perplexity
        # total_bleu += bleu_score
        # wandb.log({"validation_loss": val_loss, "validation perplexity": val_perplexity, "bleu_score": bleu_score})
    
        # Save the model after each epoch
        

    print(f"Total training loss: {total_train_loss}")

    print(f"Total training perplexity: {total_train_perplexity}")
    
    print(f"Total BLEU score: {total_bleu}")
    #print(f"Total ROUGE score: {total_rouge}")
    wandb.log({"total training loss": total_train_loss, "total training perplexity": total_train_perplexity})
    # Log the average training and validation loss for the epoch
    average= {"average training loss": total_train_loss / (epoch + 1), "average training perplexity": total_train_perplexity / (epoch + 1), "average bleu score": total_bleu / (epoch + 1)}
    print(average)
    wandb.log(average)

    wandb.finish()

    # Generate text using the model
    input_text = "Inkatazakurekera  Ibyivugo Inkatazakurekera ya Rugombangogo Ndi intwali yabyirukiye gutsinda, nsinganirwa nshaka kurwana Ubwo duteye Abahunde, nywuhimbajemo intanage Intambara nyirema igihugu cy'umuhinza nakivogeye Umukinzi ampingutse imbere n'isuli,"
    predicted_text = generate_text(model, tokenizer, input_text)
    print(f"Input text: {input_text}")

    print(f"Predicted text: {predicted_text}")


if __name__ == "__main__":
    main()