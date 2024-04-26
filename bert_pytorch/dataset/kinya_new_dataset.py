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
    def __init__(self, corpus_path, vocab, seq_len, encoding="ISO-8859-1", corpus_lines=None, on_memory=True,is_inference=False):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)

        self.common_english_words = set(words.words())
        
        #FUll path of the corpus file excluding the extension
        corpus_path_file_name = os.path.splitext(corpus_path)[0]
       
    
        corpus_path_file_name, _ = os.path.splitext(corpus_path)
        output_path = f"{corpus_path_file_name}_preprocessed.txt"
        final_output_path = f"{corpus_path_file_name}_final.txt"
        # preprocess_corpus if the final file does not exist or if it is empty
        if is_inference:
            preprocess_corpus(corpus_path, output_path, final_output_path)
        else:
            if not os.path.exists(final_output_path) or os.stat(final_output_path).st_size == 0:
                preprocess_corpus(corpus_path, output_path, final_output_path)

            else:
                print("Final file exists")

        with open(final_output_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(final_output_path, "r", encoding=encoding)
            self.random_file = open(final_output_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab["[CLS]"]] + t1_random + [self.vocab["[SEP]"]]
        t2 = t2_random + [self.vocab["[SEP]"]]

        t1_label = [self.vocab["[PAD]"]] + t1_label + [self.vocab["[PAD]"]]
        t2_label = t2_label + [self.vocab["[PAD]"]]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab["[PAD]"] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.get(token, self.vocab["[UNK]"])

                output_label.append(self.vocab.get(token, self.vocab["[UNK]"]))

            else:
                tokens[i] = self.vocab.get(token, self.vocab["[UNK]"])
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]