import numpy as np
import torch 
import torch.nn as nn
import h5py as h
import torch.nn.functional as F

class cVAE(nn.Module):
    def __init__(self,input_shape,
                      latent_dim,
                      hidden_dim=None,
                      n_channels=100,
                      kernel=5):
        super().__init__()

        self.z_dim = latent_dim
        self.input_shape = input_shape
        self.n_channels = n_channels

        test_data = torch.rand((1,1,*input_shape))

        self.encoder = nn.Sequential(
            nn.Conv2d(1,n_channels,kernel,stride=1,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channels,n_channels,kernel,stride=1,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channels,n_channels,kernel,stride=2,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channels,1,kernel,stride=1,padding=1),
            nn.LeakyReLU(0.2),
        )
        self.flat_layer = nn.Flatten()
        if hidden_dim is None:
            self.final_size = self.encoder(test_data).shape[1:]
            self.hidden = self.flat_layer(
                                self.encoder(test_data)
                                    ).shape[-1]
        self.mean = nn.Linear(self.hidden,self.z_dim)
        self.logvar = nn.Linear(self.hidden,self.z_dim)
        self.fc = nn.Linear(self.z_dim,self.hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1,n_channels,kernel,stride=1,padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channels,n_channels,kernel,stride=2,padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channels,n_channels,kernel,stride=1,padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channels,1,kernel,stride=1,padding=1),
            #nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def encode(self,x):
        x = self.encoder(x)
        x = self.flat_layer(x)
        mu = self.mean(x) 
        lv = self.logvar(x)
        z = self.sample(mu,lv)
        return z,mu,lv

    def decode(self,z):
        z = self.fc(z)
        z = z.view(-1,*self.final_size)
        output = self.decoder(z)
        return output

    def forward(self,x):
        z,mu,lv = self.encode(x)
        recon = self.decode(z)
        recon = recon.reshape(-1,*self.input_shape)
        return recon,mu,lv,z

    def embed(cmap):
        return encode(cmap)[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = h.File("/mnt/data2/cov_sims/cov_cmaps_hdf.h5","r")
    data = test_data["analysis/pooled_25"]
    test_data = torch.tensor(data[0]).view(1,1,23,23).to(device).float()
    test_shape = data[0].shape

    vae = cVAE(test_shape,10).to(device)
    res = vae(test_data)
    print(res[0].shape)
