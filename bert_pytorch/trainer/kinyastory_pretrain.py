import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
from .optim_schedule import ScheduledOptim
sys.path.append('../')
from model import BERTLM, BERT

import tqdm
import logging


class KinyaStoryBERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        logging.basicConfig(level=logging.INFO)

        logging.info(f'Initialized BERT trainer with cuda: {cuda_condition}, device: {self.device}')
        logging.info(f'Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}')

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    
    
    def iteration(self, epoch, data_loader, train=True):
        
        logging.info(f'Starting iteration, epoch: {epoch}, train: {train}')
    
        str_code = "train" if train else "test"
    
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
    
        avg_loss = 0.0
    
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            logging.info(f'bert_input shape: {data["bert_input"].shape}, device: {data["bert_input"].device}')
            logging.info(f'segment_label shape: {data["segment_label"].shape}, device: {data["segment_label"].device}')
            logging.info(f'model device: {next(self.model.parameters()).device}')
    
            # forward the masked language model
            mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            logging.info(f'Forwarded masked language model, output: {mask_lm_output}')
    
            # NLLLoss of predicting masked token word
            try:
                mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            except Exception as e:
                logging.error(f'Error in model forward: {e}')
                raise

            logging.info(f'Calculated NLLLoss, loss: {mask_loss}')
    
            # backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                mask_loss.backward()
                self.optim_schedule.step_and_update_lr()
    
            avg_loss += mask_loss.item()

            logging.info(f'Updated avg_loss: {avg_loss}')
    
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": mask_loss.item()
            }
    
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                logging.info(f'Iteration: {i}, loss: {mask_loss.item()}, avg_loss: {avg_loss / (i + 1)}')
    
        logging.info(f'Finished iteration, epoch: {epoch}, avg_loss: {avg_loss / len(data_iter)}')
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        str_code = "train" if train else "test"
    
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
    
        avg_loss = 0.0
    
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
    
            # forward the masked language model
            mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
    
            # NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
    
            # backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                mask_loss.backward()
                self.optim_schedule.step_and_update_lr()
    
            avg_loss += mask_loss.item()
    
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": mask_loss.item()
            }
    
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
    
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
