import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            conv(1, 4), 
            encode(4), 
            nn.MaxPool2d(2),
            encode(8), 
            nn.MaxPool2d(2),
            encode(16), 
            nn.MaxPool2d(2),
            encode(32), 
            nn.MaxPool2d(2),
            nn.Flatten(),
            flat_layer(128, 32),
            flat_layer(32, 8),
            flat_layer(8, 3), 
            nn.Softmax(dim = 1)
        )

    def forward(self, img):
        return self.model(img)
    

def conv(n_in, n_out, kernel_size=3): 
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size, padding="same", padding_mode="reflect"),
        nn.BatchNorm2d(n_out),
        nn.ReLU()
    )

def encode(n): return nn.Sequential(conv(n, 2 * n), conv(2 * n,  2 * n))

def flat_layer(n_in, n_out): 
    return nn.Sequential(
        nn.Linear(n_in, n_out),
        nn.ReLU()
    )
    
def load_encoder(fname):
    encoder = Encoder()
    encoder.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    encoder.eval()
    return encoder