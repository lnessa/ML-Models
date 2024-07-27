from Utility.utility import handle_args
from .train import train_unet
from .demo import unet_demo

args_dict = {
    "train_unet": train_unet, 
    "unet_demo": unet_demo
}

if __name__ == "__main__": handle_args(args_dict, "UNET")