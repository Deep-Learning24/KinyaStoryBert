import os
import pickle
import re
import sys
sys.path.append("../")

# Define special tokens
special_tokens = ["[PAD]", "[CLS]", "[MASK]", "[SEP]", "[UNK]"]

def read_datasets(directory):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="ISO-8859-1") as file:
                datasets.append(file.read())
    return datasets

def tokenize(text):
    # Construct the regular expression pattern dynamically
    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    pattern = r"\b\w+(?:'\w+)?\b|" + special_tokens_pattern + r"|[\.,!?;:]"
    
    # Tokenize using the constructed pattern
    tokens = re.findall(pattern, text)
    return tokens

def create_vocabulary(datasets):
    all_tokens = []
    for dataset in datasets:
        tokens = tokenize(dataset)
        all_tokens.extend(tokens)
    vocabulary = {}
    index = 0
    for special_token in special_tokens:
        vocabulary[special_token] = index
        index += 1
    for token in all_tokens:
        if token not in vocabulary:
            vocabulary[token] = index
            index += 1
    return vocabulary

def save_vocabulary(vocabulary, filename):
    with open(filename, "wb") as file:
        pickle.dump(vocabulary, file)

def load_vocabulary(filename):
    with open(filename, "rb") as file:
        vocabulary = pickle.load(file)
    return vocabulary
    
def encode_text(text, vocabulary):
    # Tokenize the text
    tokens = tokenize(text)

    # Encode tokens using the vocabulary
    encoded_text = []
    for token in tokens:
        if token in vocabulary:
            encoded_text.append(vocabulary[token])
        else:
            encoded_text.append(vocabulary["[UNK]"])  # Handle unknown tokens
    return encoded_text

def decode_indices(indices, vocabulary,skip_special_tokens=True):
    decoded_text = []
    for index in indices:
        if index in vocabulary.values():
            # Find the token corresponding to the index in the vocabulary
            token = next(key for key, value in vocabulary.items() if value == index)
            if skip_special_tokens and token in special_tokens:
                continue
            decoded_text.append(token)
        else:
            decoded_text.append("[UNK]")  # Handle unknown indices
    return decoded_text

if __name__ == "__main__":


    # Read datasets
    datasets_directory = "kinyastory_data"
    datasets = read_datasets(datasets_directory)

    # Create vocabulary
    vocabulary = create_vocabulary(datasets)

    # Save vocabulary
    vocabulary_filename = "dataset/vocabulary.pkl"
    save_vocabulary(vocabulary, vocabulary_filename)

    # Example usage of encoder and decoder functions
    text = "Ndi intwali yabyirukiye gutsinda, nsinganirwa nshaka kurwana n'abandi bantu."
    encoded_text = encode_text(text, vocabulary)
    decoded_text = decode_indices(encoded_text, vocabulary)

    print("Encoded text:", encoded_text)
    print("Decoded text:", decoded_text)
