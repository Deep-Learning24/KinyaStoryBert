import argparse
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large")

# model = AutoModelForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large")

# # Load the model
# # model = load_model(model,)

# input_text = "Ejo ndikwiga nagize abashyitsi baje kunsura. Ndashaka kubona niba bazakwiga cyangwa se bazasura. [MASK] ni umwihariko w'abashyitsi."
# encoded_input = tokenizer(input_text, return_tensors='pt',truncation=True, padding='max_length', max_length=128)

# # Create the labels by shifting the input_ids to the right
# labels = encoded_input["input_ids"].clone()

# # Shift the inner list to the right
# labels[0, :-1] = encoded_input["input_ids"][0, 1:]


# # Set the first token of the inner list to -100
# labels[0, -1] = tokenizer.pad_token_id


# # Set the last token of the inner list to the pad_token_id
# labels[:, -1] = -100


# # Squeeze the tensors and return them
# inputs = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
# inputs["labels"] = labels.squeeze(0)

# # # Print the input dictionary
# # print("Input dictionary:")
# # print(inputs)
# print(encoded_input.keys())

# print(encoded_input)
# output = model(**encoded_input)
# print("Keys:", output.keys())
# print("Output shape:", output.logits.shape)
# print(output)



# # Get the logits from the model's output
# logits = output.logits
# loss = output.loss
# print("Loss:", loss)



# # Convert the labels to tensor
# labels = torch.tensor(labels)

# # Initialize the loss function
# loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# # Compute the loss
# loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

# print("Loss:", loss.item())
# # # Get the token IDs of the most probable tokens
# predicted_token_ids = torch.argmax(logits, dim=-1)

# # # Convert the token IDs back into strings
# predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0])
# print("Predicted tokens:", predicted_tokens)

# # Join the tokens into a single string
# predicted_text = ' '.join(predicted_tokens)
# print("Predicted text:", predicted_text)

# # Replace the masked tokens in the input text with the predicted tokens
# for i, token in enumerate(predicted_tokens):
#     if token == "[MASK]":
#         input_text = input_text.replace("[MASK]", predicted_tokens[i], 1)

# print("Generated text:", input_text)

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

def load_model(model, model_path, device='cpu'):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.to(device)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model path {model_path} does not exist")
        model.to(device)
        return model

def generate_text(model, tokenizer, input_text, max_len=128):
    encoded_input = tokenizer(input_text, return_tensors='pt',truncation=True, padding='max_length', max_length=max_len)
    encoded_input = {key: tensor.to(model.device) for key, tensor in encoded_input.items()}
    output = model(**encoded_input)
    logits = output.logits
    predicted_index = torch.argmax(logits, dim=-1)
    predicted_text = tokenizer.decode(predicted_index[0])
    # Find the NNL loss of the predicted text
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), encoded_input["input_ids"].view(-1))

    return predicted_text, loss.item()

class KinyaBertInference:

    def __init__(self, model_path,lasy_saved_epoch, device):
        self.model = AutoModelForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large")
        self.model = load_model(self.model, f"{model_path}_epoch_{lasy_saved_epoch}.pth", device)
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    def generate_text(self, input_text):
        generated_text,loss = generate_text(self.model, tokenizer, input_text)
        # Calculate the Blue and perplexity of the generated text
        bleu = calculate_bleu(input_text, generated_text)
        perplexity = calculate_perplexity(loss)

        print("Loss:", loss)

        return generated_text, bleu, perplexity
        
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
    parser.add_argument("-o", "--output_path", default = "bert_pytorch/output/bert.model_finetuned" , type=str, help="ex)output/bert.model_finetuned")
    parser.add_argument("-p", "--last_saved_epoch", type=int, default=None, help="epoch of last saved model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--text", type=str, default="Umusaza n'abuzukuru be Umugani muremure Bawucira abana babigisha ubwumvikane mumuryango Umusaza yari afite	abuzukuru batatu b'abasore; abo bana ntibumvikane, ahubwo iteka bagahora batongana. Sekuru yabireba bikamubabaza cyane. Bukeye arababwira ati bana banjye, mujye mubana neza	ntimugahore mupfa ubusa. Abasore bawe ntibabyiteho bikomereza umwiryane wabo.", help="text to generate")
    
    
    args = parser.parse_args()
    # print(args)
    print("Original text:", args.text)
    kinyaBertInference = KinyaBertInference(args.output_path,args.last_saved_epoch, args.device)
    generated_text, bleu, perplexity = kinyaBertInference.generate_text(args.text)
    print("Generated text:", generated_text)
    print("BLEU score:", bleu)
    print("Perplexity:", perplexity)

if __name__ == "__main__":
    main()