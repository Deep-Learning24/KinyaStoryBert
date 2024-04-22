from torch.utils.data import Dataset
import torch

class KinyaStoryBertDataset(Dataset):
    def __init__(self, vocab, seq_len, tokenized_data_file_path = '../tokenized_data.pt'):
        self.vocab = vocab
        self.seq_len = seq_len
        self.tokenized_data_file_path = tokenized_data_file_path
        
        self.tokenized_data = torch.load(self.tokenized_data_file_path)

    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        tokenized_sequence = self.tokenized_data[idx]
    
        # Use special tokens from your tokenizer
        bert_input = [self.vocab["[CLS]"]] + tokenized_sequence[0] + [self.vocab["[SEP]"]]
        bert_label = bert_input[1:] + [self.vocab["[PAD]"]]  # shift input to the right to create labels
    
        segment_label = ([1 for _ in range(len(bert_input))])[:self.seq_len]
        bert_input = bert_input[:self.seq_len]
        bert_label = bert_label[:self.seq_len]
    
        padding = [self.vocab["[PAD]"] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
    
        # Generate is_next label
        # Assuming tokenized_sequence[2] is a boolean indicating whether the second sentence is the next sentence
        is_next = [1] if tokenized_sequence[2] else [0]
    
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next}
    
        return {key: torch.tensor(value) for key, value in output.items()}