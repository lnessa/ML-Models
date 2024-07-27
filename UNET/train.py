from Data.DataLoader import HeatMaps
from .UNET import UNET
from .loss import unet_loss
from Utility.train import train

def train_unet():
    test_dataset = HeatMaps('test')
    train_dataset = HeatMaps('train')
    batch_size = 50
    model_name = "unet1"
    train(UNET, unet_loss, model_name, train_dataset, test_dataset, batch_size)




