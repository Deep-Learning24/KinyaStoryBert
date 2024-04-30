import os
import re
from torch.utils.data import Dataset
import tqdm
import torch
import random
from transformers import AutoTokenizer
import nltk


from nltk.corpus import words

nltk.download('words')


def preprocess_corpus(corpus_path, output_path, final_output_path):
    # First pass: Remove empty lines and write to temporary file
    with open(corpus_path, "r", encoding="ISO-8859-1") as f:
        with open(output_path, "w", encoding="ISO-8859-1") as f_out:
            for line in tqdm.tqdm(f, desc="Preprocessing Dataset"):
                # Check if the line is empty or contains only whitespace
                if line.strip():  # Check if line is not empty
                    f_out.write(line)

    # Second pass: Split lines into two parts and add a tab between them
    with open(output_path, "r", encoding="ISO-8859-1") as f:
        with open(final_output_path, "w", encoding="ISO-8859-1") as f_out:
            for line in tqdm.tqdm(f, desc="Preprocessing Dataset again"):
                line = line.strip()  # Remove leading/trailing whitespace
                if line:
                    words = line.split()
                    line_middle = len(words) // 2
                    # If the middle word is [SEP], split after it
                    if "[SEP]" in words:
                        index = words.index("[SEP]")
                        parts = [" ".join(words[:index + 1]), " ".join(words[index + 1:])]
                    else:
                        parts = [" ".join(words[:line_middle]), " ".join(words[line_middle:])]
                    f_out.write(parts[0] + "\t" + parts[1] + "\n")

    # Remove the temporary file
    os.remove(output_path)

class KinyaStoryNewDataset(Dataset):
    def __init__(self, file, tokenizer, seq_len=128, on_memory=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.file = file
        self.data = []

        # Load the data into memory
        if self.on_memory:
            with open(file, "r", encoding="ISO-8859-1") as f:
                for line in f:
                    self.data.append(line.strip())
        else:
            self.file = open(file, "r", encoding="ISO-8859-1")
        
        corpus_path_file_name = os.path.splitext(file)[0]
        
        output_path = f"{corpus_path_file_name}_preprocessed.txt"
        final_output_path = f"{corpus_path_file_name}_final.txt"
        preprocess_corpus(file, output_path, final_output_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the text for this index
        if self.on_memory:
            text = self.data[index]
        else:
            self.file.seek(index)
            text = self.file.readline().strip()

        # Tokenize the text and create the input dictionary
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.seq_len, return_tensors='pt')

        # Create the labels by shifting the input_ids to the right
        labels = inputs["input_ids"].clone()
        labels[:-1] = inputs["input_ids"][1:]
        labels[-1] = self.tokenizer.pad_token_id

        # Squeeze the tensors and return them
        inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}
        inputs["labels"] = labels.squeeze(0)
        return inputs