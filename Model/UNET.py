import torch.nn as nn
import torch
from functools import reduce

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        layers = [16, 8, 4]

        core = reduce(nest, layers, conv(32, 32))
        self.model = nn.Sequential(conv(1, 4), core, nn.Conv2d(4, 4, 1), nn.Softmax2d())

    def forward(self, img):
        return self.model(img)
    

def nest(mdl, n): 
    return nn.Sequential(
        encode(n), 
        SkipConnection(nn.Sequential(downscale(), mdl, upscale(n))),
        decode(n)
    )

def conv(n_in, n_out, kernel_size=3): 
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size, padding="same", padding_mode="reflect"),
        nn.BatchNorm2d(n_out),
        nn.ReLU()
    )

def encode(n): return nn.Sequential(conv(n, 2 * n), conv(2 * n,  2 * n))
def decode(n): return nn.Sequential(conv(4 * n,  2 * n), conv(2 * n, n))

def downscale(): 
    return nn.MaxPool2d(2)

def upscale(n): 
    return nn.ConvTranspose2d(2 * n, 2 * n, 2, stride=2)

class SkipConnection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        rest = self.model(X)
        x_diff = abs(X.shape[-2] - rest.shape[-2])
        y_diff = abs(X.shape[-1] - rest.shape[-1])
        rest = torch.nn.functional.pad(rest, [0, y_diff, 0, x_diff], value = 0)
        return torch.cat((X, rest), 1)       
    
def load_unet(fname):
    unet = UNET()
    unet.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    unet.eval()
    return unet