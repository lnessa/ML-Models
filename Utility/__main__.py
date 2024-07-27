from .utility import handle_args
from .demo import ncut_loss_demo

args_dict = {
    "ncut_loss_demo": ncut_loss_demo
}

if __name__ == "__main__": handle_args(args_dict, "Utility")