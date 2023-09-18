import os
import pickle
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, logging

import wandb
#from modules import data, earlystop, loss, models, optimisers, wandb_log
import data, earlystop, loss, models, optimisers, wandb_log

# rank = identifier for each 'process', world_size = total number of processes
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # this MASTER machine co-ordinates the distribution
    os.environ['MASTER_PORT'] = '12358'

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)


class Model():
    def __init__(self,
                 model_link,
                 data_path,
                 rank,
                 project,
                 tags,
                 config,
                 ddp,
                 log):
        torch.cuda.empty_cache()
        logging.set_verbosity_error()
        self.rank = rank
        self.model_link = model_link
        self.device = torch.device(f'cuda:{self.rank}')
        self.log = True if log and self.rank == 0 else False
        self.ddp = ddp
        self.early_stopper = earlystop.EarlyStopper(patience=5)
        self.hidden_states, self.hidden_labels = {}, {}

        #model, tokenizer, data, logger
        self.model = models.Model_Base(self.model_link).model
        print('-- Model Loaded.')
        #self.model = models.ActivationHook(self.model, layers=[n for n,_ in self.model.named_modules() if 'roberta.encoder' in n])
        self.model = self.model.to(self.device)
        if ddp:
            self.model = DDP(self.model, device_ids=[self.rank])
        self.tokenizer = AutoTokenizer.from_pretrained(model_link)
        print('-- Tokenizer Loaded.')
        self.data = data.DataManager(data_path, self.tokenizer)
        print('-- Data Loaded.')

        if self.log:
            self.logger = wandb_log.WandBLogger(rank=self.rank,
                                                project=project,
                                                tags=tags,
                                                config=config,
                                                model_link=model_link)
        if self.rank == 0:
            self.createModelDir()

        # config unpacking
        self.num_epochs = config.get('num_epochs', 1)
        self.batch_size = config.get('batch_size', 32)
        self.train_split = config.get('train_split', 0.7)
        self.lr = config.get('lr', 0.01)
        self.loss_fn = config.get('loss_fn', torch.nn.MSELoss)()
        self.optim = config.get(
            'optimiser', optimisers.Optim_BaseAdamW)(self.model, self.lr)
        self.lr_scheduler = config.get(
            'scheduler', torch.optim.lr_scheduler.LinearLR)
        self.num_warmup_steps = config.get('num_warmup_steps', 0)
        self.reinit_n = config.get('reinit_n', 0)
        self.split_method = config.get('split_method', 'random')
        if self.reinit_n > 0:
            self.reinit_weights(self.reinit_n)
        self.comp_smiles = config.get('comp_smiles', False)

        if self.lr_scheduler == torch.optim.lr_scheduler.LinearLR:
            self.lr_scheduler = self.lr_scheduler(
                self.optim.optimiser, start_factor=1.0, end_factor=0, total_iters=self.num_epochs)
        elif self.lr_scheduler == torch.optim.lr_scheduler.ExponentialLR:
            gamma = (1/1000)**(1/self.num_epochs)
            self.lr_scheduler = self.lr_scheduler(
                self.optim.optimiser, gamma=gamma)
        elif self.lr_scheduler == transformers.get_linear_schedule_with_warmup:
            self.lr_scheduler = self.lr_scheduler(
                self.optim.optimiser, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_epochs)

        # build dataloaders
        if self.comp_smiles:
            self.data.process_comparativeSMILES()
        self.data.getTrainTestSplit(self.train_split, self.split_method)
        self.train_loader, self.test_loader = self.data.buildDataLoader(
            self.batch_size, ddp)
        print('-- DataLoaders Created.')

    def reinit_weights(self, reinit_n):
        layer_count = 0
        if self.ddp:
            for n, m in list(self.model.module.named_modules()):
                if layer_count < reinit_n:
                    if 'embeddings.' in n or 'encoder.' in n:
                        layer_count += 1
                        self.model.module._init_weights(m)
            print(f'-- Re-initialised {layer_count} layers')
        else:
            for n, m in list(self.model.named_modules()):
                if layer_count < reinit_n:
                    if 'embeddings.' in n or 'encoder.' in n:
                        layer_count += 1
                        self.model._init_weights(m)
            print(f'-- Re-initialised {layer_count} layers')

    def createModelDir(self):
        now = dt.now().strftime('%d-%m-%Y')
        now_ = dt.now().strftime('%d-%m-%Y-%H:%M')
        if wandb.run is None:
            self.model_dir = os.path.join('models', f'model-{now_}')
        else:
            self.model_dir = os.path.join(
                'models', f'model-{now}-{wandb.run.name}')

        if not os.path.isdir('models'):
            os.mkdir('models')
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def saveCheckpoint(self, preds, labels):
        r2_score = sklearn.metrics.r2_score(labels.cpu(), preds.cpu())
        print(f'---- R2 Score: {r2_score:.2f}')
        if r2_score > 0.6:
            if self.ddp:
                ckp = self.model.module.state_dict()
                torch.save(ckp, f'{self.model_dir}/model_state_dict.pt')
                print(f'-- Model state dict saved: {self.model_dir}')
            else:
                ckp = self.model.state_dict()
                torch.save(ckp, f'{self.model_dir}/model_state_dict.pt')
                print(f'-- Model state dict saved: {self.model_dir}')

    def loadModel(self, path):
        self.model.load_state_dict(torch.load(path))

    def getHidden(self, labels, arr: torch.tensor):
        arr = arr.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy().reshape(-1)
        return arr, labels

    def saveHidden(self, arr, labels, epoch):
        self.hidden_states[epoch] = arr
        self.hidden_labels[epoch] = labels
        with open(f'{self.model_dir}/hidden_states.pkl', 'wb') as file:
            pickle.dump(self.hidden_states, file)
        with open(f'{self.model_dir}/hidden_labels.pkl', 'wb') as file:
            pickle.dump(self.hidden_labels, file)

    def train(self):
        print('-- Starting Training:')
        print(f'''-- Parameters: 
              ---- Epochs: {self.num_epochs}
              ---- Batch Size: {self.batch_size}
              ---- Train Split: {self.train_split}
              ---- LR: {self.lr}
              ---- Optimiser: {type(self.optim)}
              ---- Loss Metric: {self.loss_fn}
              ---- LR Scheduler: {type(self.lr_scheduler)}
              ---- No. GPU: {torch.cuda.device_count()}
              ---- Split: {self.split_method}
              ---- wandb logging: {True if self.log else False}
              ''')
        print(f'-- Training on GPU {self.rank}')
        for epoch in range(self.num_epochs):
            start = time.time()
            for i, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.rank) for k, v in batch.items()}
                outputs = self.model(
                    batch['input_ids'], batch['attention_mask'], output_hidden_states=True,)
                preds = outputs.logits
                labels = batch['labels']
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optim.optimiser.step()
                self.optim.optimiser.zero_grad()

                if (i + 1) % 5 == 0:
                    print(
                        f'-- Epoch: {epoch} -- Iter: {i + 1} -- GPU: {self.rank} -- Loss: {loss}')

                if i == 0 and epoch % 5 == 0 and self.rank == 0:
                    arr, l = self.getHidden(labels, torch.stack(list(outputs.hidden_states[-3:])))
                    self.saveHidden(arr, l, epoch)

            self.lr_scheduler.step()

            preds = torch.Tensor().to(self.rank)
            labels = torch.Tensor().to(self.rank)
            for batch in self.test_loader:
                batch = {k: v.to(self.rank) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(
                        batch['input_ids'], batch['attention_mask'])
                preds = torch.cat((preds, outputs.logits.reshape(-1)), 0)
                labels = torch.cat((labels, batch['labels'].reshape(-1)), 0)

            early_stop_flag = torch.zeros(1).to(self.rank)
            test_loss = self.loss_fn(outputs.logits, batch['labels'])
            if self.rank == 0 and False:
                if self.early_stopper.early_stop(test_loss):
                    early_stop_flag += 1
            if self.ddp:
                dist.all_reduce(early_stop_flag, op=dist.ReduceOp.SUM)
            if early_stop_flag > 0:
                print(f'GPU {self.rank} - Training Stopped.')
                break

            end = time.time()
            print(f'-- Elapsed: {end - start} -- GPU: {self.rank}')

            if (epoch + 1) % 5 == 0 and self.rank == 0:
                self.saveCheckpoint(preds, labels)

            log_dict = {
                'num_epochs': self.num_epochs,
                'labels': labels.cpu(),
                'preds': preds.cpu(),
                'epoch': epoch,
                'start': start,
                'end': end,
                'train_loss': loss,
                'test_loss': test_loss,
                'lr_scheduler': self.lr_scheduler,
                'optimiser': self.optim
            }
            if self.log:
                self.logger.log(self.model_dir, **log_dict)

        if self.log:
            wandb.finish()
        del outputs, loss, batch, self.data, self.train_loader, self.test_loader
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())


def main_ddp(rank, world_size, model_link, data_path, project, tags, config, ddp, log):
    ddp_setup(rank, world_size)
    m = Model(model_link,
                  data_path,
                  rank,
                  project,
                  tags,
                  config,
                  ddp,
                  log)
    m.train()
    dist.destroy_process_group()
    
def main(model_link, data_path, project, tags, config, ddp, log):
    rank = 0
    m = Model(model_link,
                  data_path,
                  rank,
                  project,
                  tags,
                  config,
                  ddp,
                  log)
    m.train()


if __name__ == '__main__':
    config = {
        'num_epochs': [80],
        'batch_size': [128],
        'train_split': [0.85],
        'lr': [0.0000690452546751227],
        'loss_fn': [loss.RMSE_loss],
        'optimiser': [optimisers.Optim_BaseAdamW],
        'scheduler': [torch.optim.lr_scheduler.LinearLR],
        'num_warmup_steps': [10],
        # 'reinit_n': [0,5,10,20],
        'comp_smiles': [False],
        # ['alternateProteinSplit']
        'split_method': ['random', 'scaffold', 'fingerprintSimilarity', ]
    }

    project = 'sheepy-optim'
    tags = ['test_rewrite']
    model_link = 'DeepChem/ChemBERTa-10M-MTR'
    data_path = '/data/ben/bindingDB/processed_data/bindingDB_DRD2_HUMAN_P14416.csv'
    rank = 0

    m = Model(model_link,
              data_path,
              rank,
              project,
              tags,
              config,
              ddp=False,
              log=False)

    m.train()

