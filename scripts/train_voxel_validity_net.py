import os
import yaml
import pickle
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.utils.data
from torch.utils.data import TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ljcmp.models.validity_network import VoxelValidityNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')

"""pytorch lightning training code for ModelVAE"""
class VoxelValidityNetModule(pl.LightningModule):
    def __init__(self, h_dim, z_dim, x_dim=None, c_dim=0, scene_range=range(500), n_grid=32, max_config_len=1000, tag_name='no_tag',
                voxel_latent_dim=4,
                batch_size=128, lr=1e-3, exp_name='panda_dual_arm_with_fixed_orientation_condition'):
        super(VoxelValidityNetModule, self).__init__()
        
        """model parameters"""
        self.batch_size = batch_size
        self.lr = lr
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        print('z_dim', z_dim)
        print('c_dim', c_dim)
        print('h_dim', h_dim)

        self.model = VoxelValidityNet(h_dim=h_dim, z_dim=z_dim, c_dim=c_dim, voxel_latent_dim=voxel_latent_dim).to(device)

        """load dataset"""
        V0 = torch.zeros((len(scene_range), n_grid**3), dtype=torch.float32)
        D0 = torch.zeros((len(scene_range), max_config_len, z_dim), dtype=torch.float32)
        Y0 = torch.zeros((len(scene_range), max_config_len, 1), dtype=torch.float32)
        C0 = torch.zeros((len(scene_range), max_config_len, c_dim), dtype=torch.float32)

        scene_data_path = f'dataset/{exp_name}/scene_data/'
        for i in tqdm(scene_range, desc='loading scene data'):
            scene_name = 'scene_{:04d}'.format(i)
            scene = yaml.load(open(os.path.join(scene_data_path, scene_name, 'scene.yaml'), 'r'), Loader=yaml.FullLoader)
            if c_dim > 0:
                c = scene['c']
            joint_configs = pickle.load(open(os.path.join(scene_data_path, scene_name, tag_name, 'config.pkl'), 'rb'))
            config_len = min(len(joint_configs['valid_set']), len(joint_configs['invalid_set']), max_config_len//2)
            assert (len(joint_configs['valid_set']) >= config_len) and (len(joint_configs['invalid_set']) >= config_len)

            valid_set = joint_configs['valid_set'][:config_len]
            invalid_set = joint_configs['invalid_set'][:config_len]
            for j in range(config_len):
                D0[i, j, :] = torch.from_numpy(valid_set[j][0])
                Y0[i, j] = 1
                D0[i, j+config_len, :] = torch.from_numpy(invalid_set[j][0])
                Y0[i, j+config_len] = 0

                if c_dim > 0:
                    C0[i, j, :] = torch.from_numpy(valid_set[j][2])
                    C0[i, j+config_len, :] = torch.from_numpy(invalid_set[j][2])
            
            voxel = np.load(os.path.join(scene_data_path, scene_name, 'voxel.npy')).flatten()
            # numpy float to bool
            # voxel = voxel > 0.5
            V0[i, :] = torch.from_numpy(voxel)
        
        V0 = V0.to(device)
        D0 = D0.to(device)
        Y0 = Y0.to(device)
        C0 = C0.to(device)

        len_total = len(scene_range)
        train_set_len = int(len_total*0.9)
        
        """dataset with device"""
        train_dataset = TensorDataset(D0[:train_set_len], 
                                      Y0[:train_set_len],
                                      C0[:train_set_len],
                                       V0[:train_set_len])
        validation_dataset = TensorDataset(D0[train_set_len:], 
                                           Y0[train_set_len:],
                                           C0[train_set_len:],
                                           V0[train_set_len:])
        """dataloader with device"""
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1)
        
        print('train', len(self.train_loader))
        print('validation', len(self.validation_loader))


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        c = batch[2]
        v = batch[3]
        v = v.repeat_interleave(x.shape[1], dim=0)

        x = x.reshape(-1, self.z_dim)
        y = y.reshape(-1, 1)
        if self.c_dim > 0:
            c = c.reshape(-1, self.c_dim)

        if self.c_dim > 0:
            y_hat, voxel_latent = self.model(x, voxel=v, c=c)
        else:
            y_hat, voxel_latent = self.model(x, voxel=v)

        loss_estimation = self.model.loss(x, c, y, y_hat)

        self.log('train_loss_estimation', loss_estimation)

        return loss_estimation
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        c = batch[2]
        v = batch[3]
        v = v.repeat_interleave(x.shape[1], dim=0)
        
        x = x.reshape(-1, self.z_dim)
        y = y.reshape(-1, 1)
        if self.c_dim > 0:
            c = c.reshape(-1, self.c_dim)

        if self.c_dim > 0:
            y_hat, voxel_latent = self.model(x, voxel=v, c=c)
        else:
            y_hat, voxel_latent = self.model(x, voxel=v)

        loss_estimation = self.model.loss(x, c, y, y_hat)

        accuracy = torch.sum(torch.round(y_hat) == y).float() / y.shape[0]

        self.log('val_loss_estimation', loss_estimation)
        self.log('val_accuracy', accuracy)
        
        return loss_estimation
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        sheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        return [optimizer], [sheduler]

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.validation_loader

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--dataset_size', type=int, default=500)
parser.add_argument('--max_config_len', type=int, default=200)
parser.add_argument('--voxel_latent_dim', type=int, default=4)
parser.add_argument('--exp_name', '-E', type=str, default='panda_triple', help='panda_orientation, panda_dual, or panda_triple')

args = parser.parse_args()

exp_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=args.exp_name), 'r'), Loader=yaml.FullLoader)

epochs = args.epochs    

constraint_model_path = 'models/{exp_name}/{model_path}'.format(exp_name=args.exp_name, model_path=exp_info['constraint_model']['path'])
ckpt_name = constraint_model_path.split('/')[-1].split('.ckpt')[0]

"""check directory exist"""
run_index = 1
dir_base = f'wandb/checkpoints/{args.exp_name}/voxel_validity/'
while True:
    run_name = "H_{h_dim}_{dsz}_{tag}_{ckpt_name}_{run_index}".format(h_dim=exp_info['voxel_validity_model']['h_dim'], 
                                                                      dsz=args.dataset_size, tag=exp_info['constraint_model']['tag'], 
                                                                      ckpt_name=ckpt_name, run_index=run_index)
    run_path = os.path.join(dir_base, run_name)
    if not os.path.exists(run_path):
        break
    run_index += 1

os.makedirs(run_path, exist_ok=True)

wandb_logger = WandbLogger(project=args.exp_name+'_voxel_validity', name=run_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb_logger.experiment.config.update(args)

model = VoxelValidityNetModule(h_dim=exp_info['voxel_validity_model']['h_dim'], z_dim=exp_info['z_dim'], c_dim=exp_info['c_dim'], x_dim=exp_info['x_dim'],
                               scene_range=range(args.dataset_size), max_config_len=args.max_config_len, tag_name=exp_info['constraint_model']['tag'],
                               voxel_latent_dim=args.voxel_latent_dim,
                               batch_size=args.batch_size, exp_name=args.exp_name, lr=args.lr).to(device)

checkpoint_callback = ModelCheckpoint(dirpath=run_path, 
                                      filename='{epoch}-{val_accuracy:.2f}', 
                                      monitor="val_accuracy", mode="max", save_top_k=1, save_last=True)
trainer = Trainer(max_epochs=epochs, logger=wandb_logger, callbacks=[checkpoint_callback], val_check_interval=1.0)

trainer.fit(model)
