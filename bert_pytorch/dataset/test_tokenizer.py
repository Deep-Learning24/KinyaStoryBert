import sys
sys.path.append('../')
sys.path.append('./')

from custom_tokenizer import load_bpe_model, encode_text, decode_indices
from custom_tokenizer import special_tokens
from KinyaTokenizer import KinyaTokenizer, encode, decode
from transformers import AutoTokenizer
# Load BPE model
sp = load_bpe_model()

tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)

# Example usage of encoder and decoder functions
text = "Ndi intwali yabyirukiye gutsinda, nsinganirwa nshaka kurwana n'abandi bantu.?? [] {} [PAD] [CLS] [MASK] [SEP] [UNK]"
print("Text:", text)
encoded_text = encode_text(text, sp)
kinyabert_encoded_text = encode(text, tokenizer)
decoded_text = decode_indices(encoded_text, sp)
kinyabert_decoded_text = decode(kinyabert_encoded_text, tokenizer)

print("Encoded text:", encoded_text)
print("Decoded text:", decoded_text)
print("KinyaBERT Encoded text:", kinyabert_encoded_text)
print("KinyaBERT Decoded text:", kinyabert_decoded_text)