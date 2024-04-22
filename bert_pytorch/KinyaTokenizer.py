import json
from transformers import AutoTokenizer
import torch
import os

import pandas as pd

from multiprocessing import Pool
from tqdm import tqdm
import time


def encode(tokenizer, text):
    encoding = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_attention_mask=True,
    )
    return encoding['input_ids'], encoding['attention_mask']


def decode(tokenizer, encoded_tokens, skip_special_tokens=True):

    if isinstance(encoded_tokens[0], list):

        return [tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in encoded_tokens]
    else:

        return tokenizer.decode(encoded_tokens, skip_special_tokens=skip_special_tokens)


class KinyaTokenizer(object):
    def __init__(self, dataset_path):
        self.tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=512)
        self.dataset_path = dataset_path
        # self.extend_vocab()

    def is_word_new(self, args):
        word, vocab_set = args
        tokens = self.tokenizer.tokenize(word)
        if any(token not in vocab_set for token in tokens):
            return word
        else:
            return None

    def extend_vocab(self):
        # Load the dataset
        df = pd.read_csv(self.dataset_path)

        # Collect all unique words in the dataset
        words = set()
        for _, row in df.iterrows():
            story_input = row['story_input']
            story_output = row['story_output']
            words.update(story_input.split())
            words.update(story_output.split())

        # Convert the tokenizer's vocabulary to a set for faster lookup
        vocab_set = set(self.tokenizer.get_vocab().keys())

        # Find out which words are not in the tokenizer's vocabulary
        with Pool() as p:
            new_words = list(tqdm(p.imap(self.is_word_new, [(word, vocab_set) for word in words]), total=len(words)))
        new_words = [word for word in new_words if word is not None]

        # Get the last token ID in the current vocabulary
        last_token_id = max(self.tokenizer.get_vocab().values())

        # Add the new words to the tokenizer's vocabulary with new token IDs
        for word in tqdm(new_words):
            last_token_id += 1
            self.tokenizer.add_tokens([word])
            self.tokenizer.vocab[word] = last_token_id

        # Save the tokenizer
        self.tokenizer.save_pretrained('./kinyatokenizer')

        # Get the tokenizer configuration
        tokenizer_config = self.tokenizer.get_vocab()

        # Save the tokenizer configuration to a JSON file
        import json
        with open('tokenizer_config.json', 'w') as f:
            json.dump(tokenizer_config, f)

    def tokenize_dataset(self, max_length=512):
        df = pd.read_csv(self.dataset_path)
        tokenized_data = []
        for _, row in df.iterrows():
            story_input = str(row['story_input'])
            story_output = str(row['story_output'])

            story = story_input + ' [SEP] ' + story_output

            # Divide the story into chunks of max_length tokens
            story_chunks = [story[i:i + max_length] for i in range(0, len(story), max_length)]

            # Encode the chunks
            for chunk in story_chunks:
                # Concatenate the input and output chunks with a [SEP] token in between
                # chunk = '[CLS] ' + chunk if not chunk.startswith('[CLS] ') else chunk
                # chunk = chunk + ' [SEP]' if not chunk.endswith(' [SEP]') else chunk
                tokenized_sequence = encode(self.tokenizer, chunk)
                # Check the length of the tokenized sequence
                tokenized_data.append(tokenized_sequence)

        # Save the tokenized data
        torch.save(tokenized_data, 'tokenized_data.pt')

        return tokenized_data

    def print_sample_tokenized_data(self, tokenized_data):
        # Check the type of the tokenized data
        print("The type of the tokenized data is:", type(tokenized_data))
        # Convert the tokenized data to a list if not already
        if not isinstance(tokenized_data, list):
            tokenized_data = [tokenized_data]

        for tokenized_sequence in tokenized_data:
            # Check the shape of the tokenized sequence
            print("The type of the tokenized data is:", type(tokenized_sequence))
            input_ids, attention_mask = tokenized_sequence  # Unpack the tuple

            decoded_sequence = decode(self.tokenizer, input_ids)
            print(decoded_sequence)
            print("Length of the sequence:", len(input_ids))
            print("Length of decoded sequence:", len(decoded_sequence))


if __name__ == "__main__":
    KinyaTokenizer = KinyaTokenizer('kinyastory_data/kinyastory.csv')
    tokenized_data = KinyaTokenizer.tokenize_dataset()
    print("Tokenized data saved as tokenized_data.pt")
    first_input_ids,first_input_masks = tokenized_data[0]
    print(first_input_ids)
    print("Length of the input_ids:", len(first_input_ids))
    print(first_input_masks)
    print("Length of the attention_mask:", len(first_input_masks))
    KinyaTokenizer.print_sample_tokenized_data(tokenized_data[:10])
    print("Sample tokenized data printed")
