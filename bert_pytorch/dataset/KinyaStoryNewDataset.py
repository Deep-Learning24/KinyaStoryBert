import os
import re
from typing import Tuple
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
                    # Remove non-decodable tokens from the line
                    line = line.encode("ascii", errors="ignore").decode("ascii")
                    # Remove numbers from the line
                    line = re.sub(r'\d+', '', line)
                    f_out.write(line)

    # ... (rest of the method)

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
    def __init__(self, file, tokenizer, seq_len=128, on_memory=True,corpus_lines=None):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.file = file
        self.data = []
        self.corpus_lines = corpus_lines
    
        corpus_path_file_name = os.path.splitext(file)[0]
    
        output_path = f"{corpus_path_file_name}_preprocessed.txt"
        final_output_path = f"{corpus_path_file_name}_final.txt"
        preprocess_corpus(file, output_path, final_output_path)
    
        with open(final_output_path, "r", encoding="ISO-8859-1") as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = 0  # Initialize to 0 if None
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(final_output_path, "r", encoding="ISO-8859-1")
            self.random_file = open(final_output_path, "r", encoding="ISO-8859-1")

            for _ in range(random.randint(0,self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()


    def __len__(self):
        return self.corpus_lines

        
    def __getitem__(self, index):
        if self.on_memory:
            line = self.lines[index]
        else:
            line = self.file.readline()[:-1]
    
        # # Tokenize the text
        # if line.strip():
        # Join the two parts of the line
        line = " ".join(line)
        #print(line)
        tokenized = self.tokenizer(line, return_tensors='pt', truncation=True, padding='max_length', max_length=self.seq_len)
        #print(tokenized)
        return tokenized
        # else:
        #     return self.__getitem__(random.randint(0, self.corpus_lines - 1))
            
                        
            

        
        
