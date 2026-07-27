import torch
import torch.nn as nn
from src.constants import *

##Encoder class
class cVAE_Encoder(nn.Module):
    """Takes the input and compress into small dimension"""
    def __init__(self, in_channel=INPUT_CHANNEL, out_channel=OUTPUT_CHANNEL, size=3, step=1, latent_dim=Z_DIM):
        super().__init__()
        ##INPUT = input_dim (central Voxel signal)
        ##OUTPUT = z_dim (latent distribution parameter like mu, logvar)

        ##Input conv layer
        self.cnn_layers = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, size), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=size, stride=step),
            nn.Conv3d(out_channel,64, kernel_size=size, stride=step), 
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=size, stride=step),
            nn.Conv3d(64, kernel_size=size, stride=step), 
            nn.ReLU(),
            nn.Linear()
        )

        self.fc_mu = nn.Linear(out_dim, z_dim)
        self.fc_logvar = nn.Linear(out_dim, z_dim)
    
    ##forward pass
    def forward(self,x,c):
        ##signal and coditional feature concatination
        xc = torch.cat([x,c], dim=1)
        h = self.net_layers(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    

if __name__ == "__main__":
    print("Code succefully executed")