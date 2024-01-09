
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from ljcmp.models.latent_model import LatentModel, LatentValidityModel

import pytorch_lightning as pl
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TSVAE(torch.nn.Module, LatentModel):
    
    def __init__(self, x_dim=0, z_dim=0, c_dim=0, 
                 encoder_layer_size=[512,512], decoder_layer_size=[512,512], 
                 beta=1.0, gamma=1000., max_capacity=25, C_stop_iter=1e5, warmup_stop_iter=1e4, 
                 activation=F.leaky_relu, 
                 null_augment=True, sample_size=100, var=1e-1, 
                 kl_loss_type='B', **kwargs):
        """
        Tangent Space Augmented VAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param sample_size: sample size for tangent space sampling the batch size will be (N * sample_size)
        """
        super(TSVAE, self).__init__()
        
        self.z_dim = z_dim
        self.activation = activation
        self.sample_size = sample_size
        self.var = var
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.given_c = torch.zeros((1, c_dim), dtype=torch.float32).to(device=device)

        self.beta = beta
        self.M_N = self.z_dim / self.x_dim
        self.beta_norm = self.beta * self.M_N
        
        self.gamma = gamma
        self.gamma_norm = gamma * self.M_N

        self.C_max = torch.Tensor([max_capacity]).to(device=device)
        self.C_stop_iter = C_stop_iter
        self.warmup_stop_iter = warmup_stop_iter
        self.num_iter = 0
        self.kl_loss_type = kl_loss_type

        """
        for inference
        """
        self.default_batch_size = 1024
        self.valid_sample_storage = []

        self.null_augment = null_augment
        
        self.fc_enc = nn.ModuleList()
        for i in range(len(encoder_layer_size)):
            if i == 0:
                self.fc_enc.append(nn.Linear(x_dim + c_dim, encoder_layer_size[i]).to(device=device))
            else:
                self.fc_enc.append(nn.Linear(encoder_layer_size[i-1] + c_dim, encoder_layer_size[i]).to(device=device))
        
        self.fc_dec = nn.ModuleList()
        for i in range(len(decoder_layer_size)):
            if i == 0:
                self.fc_dec.append(nn.Linear(z_dim + c_dim, decoder_layer_size[i]).to(device=device))
            else:
                self.fc_dec.append(nn.Linear(decoder_layer_size[i-1] + c_dim, decoder_layer_size[i]).to(device=device))

        self.fc_mean = nn.Linear(encoder_layer_size[-1], z_dim).to(device=device)
        self.fc_var =  nn.Linear(encoder_layer_size[-1], z_dim).to(device=device)

        self.fc_logits = nn.Linear(decoder_layer_size[-1], x_dim).to(device=device)

    def encode(self, x, c=None):
        if c is None and self.c_dim != 0:
            c = self.given_c.repeat(x.shape[0], 1)

        if self.c_dim != 0:
            for fc in self.fc_enc:
                x = self.activation(fc(torch.concat([x, c], dim=1)))
        else:
            for fc in self.fc_enc:
                x = self.activation(fc(x))

        z_mean = self.fc_mean(x)
        z_var = F.softplus(self.fc_var(x))
        # z_var = torch.clamp(z_var, min=1e-10, max=1e10)
        
        return z_mean, z_var
        
    def decode(self, z, c=None):
        if c is None and self.c_dim != 0:
            c = self.given_c.repeat(z.shape[0], 1)
        
        if self.c_dim != 0:
            for fc in self.fc_dec:
                z = self.activation(fc(torch.concat([z, c], dim=1)))
        else:
            for fc in self.fc_dec:
                z = self.activation(fc(z))
                
        x = self.fc_logits(z)
        return x
        
    def reparameterize(self, z_mean, z_var):
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))

        return q_z, p_z

    def forward(self, x, c=None,null_basis=None,eval=False):
        if not eval and self.null_augment:
            assert(null_basis is not None)

            x_s = x.repeat_interleave(self.sample_size, dim=0)
            if self.c_dim > 0 and c is not None:
                c = c.repeat_interleave(self.sample_size, dim=0)
            nulls = null_basis.repeat_interleave(self.sample_size, dim=0)
            epsilon = torch.randn([x_s.shape[0],self.z_dim,1], device=device) * self.var
            
            x_s = x_s + torch.matmul(nulls, epsilon).squeeze(-1)
            x = x_s

        z_mean, var = self.encode(x, c)
        q_z, p_z = self.reparameterize(z_mean, var)
        z = q_z.rsample()
        x_recon = self.decode(z, c)

        return (z_mean, var), (q_z, p_z), z, x, x_recon

    def loss(self, x, x_recon, q_z, p_z):
        loss_recon = nn.MSELoss(reduction='none')(x, x_recon).sum(-1).mean()
        loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        
        r = min(self.num_iter/self.warmup_stop_iter, 1)
        C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        # thanks to https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        if self.kl_loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = loss_recon + r * self.beta_norm * loss_kl
        elif self.kl_loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            loss = loss_recon + r * self.gamma_norm * (loss_kl - C).abs()
        else: 
            raise NotImplementedError
        
        return {'loss':loss, 'loss_recon':loss_recon, 'loss_kl':loss_kl, 'C':C, 'R':r}


    def to_latent(self, q):
        single_dim = False
        if type(q) == list:
            q = np.array(q)
        if q.ndim == 1:
            q = q[None, :]
            single_dim = True
        
        tq = torch.from_numpy(q).to(device=device, dtype=torch.float32)
        z, _ = self.encode(tq)
        nz = z.detach().cpu().numpy()

        if single_dim:
            return nz[0]
        return nz

    def to_state(self, z):
        single_dim = False
        if type(z) == list:
            z = np.array(z)
        if z.ndim == 1:
            z = z[None, :]
            single_dim = True

        tz = torch.from_numpy(z).to(device=device, dtype=torch.float32)
        q = self.decode(tz)
        nq = q.double().detach().cpu().numpy().astype(np.double)
        
        if single_dim:
            return nq[0]
        return nq

    def set_condition(self, c):
        if self.c_dim == 0:
            print("Warning: no condition is needed for this model")
            return
        self.given_c = torch.from_numpy(c).to(device=device, dtype=torch.float32)

    def sample(self, num_samples, c=None):
        z = torch.normal(mean=torch.zeros([num_samples, self.z_dim]), std=torch.ones([num_samples, self.z_dim])).to(device=device)
        q = self.decode(z, c)

        nz = z.detach().cpu().double().numpy() 
        nq = q.detach().cpu().double().numpy()

        return nq, nz

    def sample_with_estimated_validity(self, num_samples:int, model:LatentValidityModel):
        while len(self.valid_sample_storage) < num_samples:
            z = torch.normal(mean=torch.zeros([self.default_batch_size, self.z_dim]), std=torch.ones([self.default_batch_size, self.z_dim])).to(device=device)

            y = model.predict(z)
            """torch get indices that y > model.threshold are True"""
            valid_indices = torch.nonzero(y > model.threshold)[:,0]

            """add tensor to list"""
            self.valid_sample_storage += z[valid_indices].detach().cpu().double().numpy().tolist()

        samples = np.array(self.valid_sample_storage[:num_samples], dtype=np.double)
        self.valid_sample_storage = self.valid_sample_storage[num_samples:]
        return samples

    def sample_with_estimated_validity_with_q(self, num_samples:int, model:LatentValidityModel):
        while len(self.valid_sample_storage) < num_samples:
            z = torch.normal(mean=torch.zeros([self.default_batch_size, self.z_dim]), std=torch.ones([self.default_batch_size, self.z_dim])).to(device=device)
            
            y = model.predict(z)
            """torch get indices that y > model.threshold are True"""
            valid_indices = torch.nonzero(y > model.threshold)[:,0]

            """add tensor to list"""
            self.valid_sample_storage += z[valid_indices].detach().cpu().double().numpy().tolist()

        samples = np.array(self.valid_sample_storage[:num_samples], dtype=np.double)
        
        self.valid_sample_storage = self.valid_sample_storage[num_samples:]
        q_samples = self.decode(torch.from_numpy(samples).to(device=device, dtype=torch.float32)).detach().cpu().double().numpy()

        return q_samples, samples 

    def sample_from_z_with_estimated_validity_with_q(self, z_input, var, num_samples:int, model:LatentValidityModel):
        z_mean = torch.from_numpy(z_input).to(device=device, dtype=torch.float32)
        z_mean = z_mean.repeat(self.default_batch_size, 1)
        std = var*torch.ones([self.default_batch_size, self.z_dim]).to(device=device, dtype=torch.float32)
        while len(self.valid_sample_storage) < num_samples:
            z = torch.normal(mean=z_mean, std=std).to(device=device)

            y = model(z)
            """torch get indices that y > model.threshold are True"""
            valid_indices = torch.nonzero(y > model.threshold)[:,0]

            """add tensor to list"""
            self.valid_sample_storage += z[valid_indices].detach().cpu().double().numpy().tolist()

        samples = np.array(self.valid_sample_storage[:num_samples], dtype=np.double)
        self.valid_sample_storage = self.valid_sample_storage[num_samples:]
        q_samples = self.decode(torch.from_numpy(samples).to(device=device, dtype=torch.float32)).detach().cpu().double().numpy()
        
        return q_samples, samples 