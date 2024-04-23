import argparse

from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset.Kinya_storydataset import KinyaStoryBertDataset
from transformers import AutoTokenizer
import os

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True,default="tokenized_data.pt", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="tokenized_val_data.pt", help="test set for evaluate train set")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")
    parser.add_argument("-p", "--last_saved_epoch", type=int, default=None, help="epoch of last saved model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)

    vocab = tokenizer.get_vocab()

    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = KinyaStoryBertDataset(vocab, seq_len=args.seq_len, tokenized_data_file_path=args.train_dataset)
    print("Train Dataset Size: ", len(train_dataset))

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = KinyaStoryBertDataset(vocab, seq_len=args.seq_len, tokenized_data_file_path=args.test_dataset) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("Train DataLoader Size: ", len(train_data_loader))

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = KinyaStoryBERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, last_saved_epoch=args.last_saved_epoch)

    print("Training Start")
    #Create output directory if it doesn't exist
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    train()