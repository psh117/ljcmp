
import yaml
import numpy as np
import torch

import torch.utils.data
from torch.utils.data import TensorDataset
import pytorch_lightning as pl

from ljcmp.models import TSVAE


torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""pytorch lightning training code for ModelVAE"""
class ModelConditionedTSVAEModule(pl.LightningModule):
    def __init__(self, x_dim, h_dim, c_dim, 
                    batch_size, lr, 
                    exp_name, max_data_len,
                    dataset_name, sample_size, **kwargs):
        super(ModelConditionedTSVAEModule, self).__init__()
                
        """model parameters"""
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = TSVAE(x_dim=x_dim, c_dim=c_dim, 
                           encoder_layer_size=[h_dim, h_dim], 
                           decoder_layer_size=[h_dim, h_dim], 
                           sample_size=sample_size, **kwargs)

        """load dataset"""
        data = np.load(f'dataset/{exp_name}/manifold/data_{dataset_name}.npy')
        nulls = np.load(f'dataset/{exp_name}/manifold/null_{dataset_name}.npy')

        if max_data_len == -1:
            max_data_len = len(data)
        
        C0 = np.array(data[:,:c_dim],dtype=np.float32)[:max_data_len]
        D0 = np.array(data[:,c_dim:],dtype=np.float32)[:max_data_len]
        N0 = np.array(nulls,dtype=np.float32)[:max_data_len]

        len_d0 = len(D0)
        self.len_d0 = len_d0
        train_set_len = int(len_d0*0.8)
        validation_set_len = len_d0 - train_set_len

        print (f'len_d0: {len_d0}, train_set_len: {train_set_len}, validation_set_len: {validation_set_len}')
        
        """dataset with device"""
        train_dataset = TensorDataset(torch.from_numpy(D0[:train_set_len]).to(device), 
                                      torch.from_numpy(N0[:train_set_len]).to(device),
                                      torch.from_numpy(C0[:train_set_len]).to(device))
        validation_dataset = TensorDataset(torch.from_numpy(D0[train_set_len:] ).to(device), 
                                           torch.from_numpy(N0[train_set_len:]).to(device),
                                           torch.from_numpy(C0[train_set_len:]).to(device))

        """dataloader with device"""
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        self.validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=validation_set_len)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_mb = batch[0]
        null_mb = batch[1]
        c_mb = batch[2]
        
        (z_mean, var), (q_z, p_z), z, x, x_recon = self.model(x_mb, c=c_mb, null_basis=null_mb if self.model.null_augment else None)

        losses = self.model.loss(x, x_recon, q_z, p_z)
        self.model.num_iter += 1
        
        self.log('train_loss', losses['loss'])
        self.log('train_recon_loss', losses['loss_recon'])
        self.log('train_KL', losses['loss_kl'])
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        x_mb = batch[0]
        null_mb = batch[1]
        c_mb = batch[2]

        (z_mean, var), (q_z, p_z), z, x, x_recon = self.model(x_mb, c=c_mb, eval=True)
        
        losses = self.model.loss(x, x_recon, q_z, p_z)
        
        self.log('val_loss', losses['loss'])
        self.log('val_recon_loss', losses['loss_recon'])
        self.log('val_KL', losses['loss_kl'])
        self.log('C', losses['C'])
        self.log('R', losses['R'])
        
        return losses['loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.validation_loader

import os
import argparse
"""pytorch lightning"""
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
"wandb logger run name includes current version choose version automatically"

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--exp_name', '-E', type=str, required=True, help='exp_name: panda_dual, panda_dual_orientation, panda_triple, or panda_orientation')
# TSA
parser.add_argument('--tsa', type=bool, default=False)
parser.add_argument('--var', type=float, default=5e-1)
parser.add_argument('--sample_size', type=int, default=100)
# VAE Model & Loss
parser.add_argument('--kl_loss_type', type=str, default='H', help='H: beta or B: gamma')
parser.add_argument('--beta', '-B', type=float, default=0.1)
parser.add_argument('--warmup_stop_iter', '-W', type=int, default=1e4)
parser.add_argument('--c_stop_iter', type=int, default=1e5) 
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--max_capacity_ratio', type=float, default=3.0) # max_capacity = max_capacity_ratio * z_dim
# Training
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dataset_name', '-D', type=str, default='10000')
parser.add_argument('--seed', '-S', type=int, default=0)
parser.add_argument('--max_data_len', '-ML', type=int, default=-1)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

x_dim = model_info['x_dim']
z_dim = model_info['z_dim']
c_dim = model_info['c_dim']
h_dim = model_info['constraint_model']['h_dim']

max_capacity = args.max_capacity_ratio * z_dim

model = ModelConditionedTSVAEModule(x_dim=x_dim, h_dim=h_dim, c_dim=c_dim, z_dim=z_dim,
                                    max_capacity=max_capacity, gamma=args.gamma, beta=args.beta,
                                    batch_size=args.batch_size, lr=args.lr, kl_loss_type=args.kl_loss_type,
                                    sample_size=args.sample_size, var=args.var, null_augment=args.tsa, 
                                    exp_name=args.exp_name, dataset_name=args.dataset_name, 
                                    C_stop_iter=args.c_stop_iter, max_data_len=args.max_data_len,
                                    warmup_stop_iter=args.warmup_stop_iter).to(device)

idx = 0
dir_base = f'wandb/checkpoints/{args.exp_name}/constraint/'
while True:
    run_name = f'V{args.var}_H{h_dim}_B{args.beta}_TY{args.kl_loss_type}_TSA{args.tsa}_D_{args.dataset_name}_ML{args.max_data_len}_{idx}'
    run_path = os.path.join(dir_base, run_name)
    if not os.path.exists(run_path):
        break
    idx += 1

np.set_printoptions(precision=3, suppress=True, linewidth=100)

os.makedirs(run_path, exist_ok=True)

wandb_logger = WandbLogger(project=args.exp_name, name=run_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb_logger.experiment.config.update(args)

checkpoint_callback = ModelCheckpoint(dirpath=run_path, 
                                      filename='{epoch}-{val_loss:.2f}-{val_recon_loss:2f}', 
                                      monitor="val_recon_loss", mode="min", save_top_k=1, save_last=True)
trainer = Trainer(max_epochs=args.epochs, logger=wandb_logger, callbacks=[checkpoint_callback])

trainer.fit(model)