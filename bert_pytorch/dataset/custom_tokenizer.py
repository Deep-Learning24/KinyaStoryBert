import os
import sentencepiece as spm
from vocab import WordVocab

# Define special tokens
special_tokens = ["[PAD]", "[CLS]", "[MASK]", "[SEP]", "[UNK]"]

def read_datasets(directory):
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="ISO-8859-1") as file:
                datasets.append(file.read())
    return datasets


def train_bpe_model(datasets, model_prefix="bpe", vocab_size= 27779):
    # Delete existing model files
    if os.path.exists(f"{model_prefix}.model"):
        os.remove(f"{model_prefix}.model")
    if os.path.exists(f"{model_prefix}.vocab"):
        os.remove(f"{model_prefix}.vocab")

    # Join all datasets into a single string
    all_text = " ".join(datasets)

    # Write all_text to a temporary file
    with open("temp.txt", "w", encoding="ISO-8859-1") as file:
        file.write(all_text)

    # Define user-defined symbols
    user_defined_symbols = list(set(special_tokens + list(".,!?';:")))

    # Train a BPE model on the text
    spm.SentencePieceTrainer.train(input="temp.txt", model_prefix=model_prefix, vocab_size=vocab_size, user_defined_symbols=user_defined_symbols)
def load_bpe_model(model_prefix="bpe"):
    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def encode_text(text, sp):
    encoded_text = sp.encode_as_ids(text)
    return encoded_text

def decode_indices(indices, sp):
    return sp.decode_ids(indices)

if __name__ == "__main__":
    # Read datasets
    datasets_directory = "kinyastory_data"
    datasets = read_datasets(datasets_directory)

    # Train BPE model
    train_bpe_model(datasets)

    # Load BPE model
    sp = load_bpe_model()

    # Example usage of encoder and decoder functions
    text = "Ndi intwali yabyirukiye gutsinda, nsinganirwa nshaka kurwana n'abandi bantu."
    encoded_text = encode_text(text, sp)
    decoded_text = decode_indices(encoded_text, sp)

    print("Encoded text:", encoded_text)
    print("Decoded text:", decoded_text)