from Utility.utility import handle_args
from .train import train_unet, train_encoder
from .demo import unet_demo, encoder_demo

args_dict = {
    "train_unet": train_unet, 
    "train_encoder": train_encoder, 
    "encoder_demo": encoder_demo,
    "unet_demo": unet_demo
}

if __name__ == "__main__": handle_args(args_dict, "Model")