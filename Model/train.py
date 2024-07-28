from Data.DataLoader import HeatMaps, Positions
from .UNET import UNET
from .Encoder import Encoder
from .loss import unet_loss, encoder_loss
from Utility.train import train

def train_unet():
    test_dataset = HeatMaps('test')
    train_dataset = HeatMaps('train')
    batch_size = 50
    model_name = "unet1"
    train(UNET, unet_loss, model_name, train_dataset, test_dataset, batch_size)

def train_encoder():
    test_dataset = Positions('test')
    train_dataset = Positions('train')
    batch_size = 50
    model_name = "encoder1"
    train(Encoder, encoder_loss, model_name, train_dataset, test_dataset, batch_size)




