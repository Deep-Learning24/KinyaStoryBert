import argparse

from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from model import BERT
from trainer import KinyaStoryBERTTrainer
from dataset import KinyaStoryNewDataset
from transformers import AutoTokenizer

import os

def freeze_or_unfreeze_bert(model,freeze_until_layer=11, freeze=False):
        """
        Freeze or unfreeze the BERT model layers ).. Gradually unfreeze the layers during training
        :param freeze_until_layer: Freeze until this layer
        :param freeze: If True, freeze the layers. If False, unfreeze the layers
        """
        layer_num = 0
        for module in model.transformer_blocks:
            if layer_num < freeze_until_layer:
                for param in module.parameters():
                    param.requires_grad = False if freeze else True
            layer_num += 1
def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True,default="kinyastory_data/train_stories.txt", type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="kinyastory_data/val_stories.txt", help="test set for evaluate train set")
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
    parser.add_argument("--is_fineturning", type=bool, default=False, help="finetuning the model")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large", max_length=128)
    
    vocab = tokenizer.get_vocab()

    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = KinyaStoryNewDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)
    print("Train Dataset Size: ", len(train_dataset))

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = KinyaStoryNewDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
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
    
    save_path = args.output_path
    if args.is_fineturning:
        trainer = KinyaStoryBERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, last_saved_epoch=args.last_saved_epoch,wandb_project_name="project-ablations", wandb_name="kinya-bert-finetuning", wandb_reinit=True)

        freeze_or_unfreeze_bert(bert, freeze_until_layer=11, freeze=True)
        save_path = "output/bert.model_finetuned"
    #wandb_project_name="project-ablations", wandb_name="kinya-bert-finetuning", wandb_reinit=True

    print("Training Start")
    #Create output directory if it doesn't exist
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_loss = float('inf')

   

    for epoch in range(args.epochs):
        if args.is_fineturning and epoch == 5:
                freeze_or_unfreeze_bert(bert, freeze_until_layer=11, freeze=False)
        trainer.train(epoch)

        if test_data_loader is not None:
            trainer.test(epoch)
             # Get the average loss for the current epoch
            current_loss = trainer.get_average_loss()
        
            # If the current loss is lower than the best loss, save the model and update the best loss
            if current_loss < best_loss:
                trainer.save(epoch, save_path)
                best_loss = current_loss

if __name__ == "__main__":
    train()