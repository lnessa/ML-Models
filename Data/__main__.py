from Utility.utility import handle_args

from .demo import loader_demo, make_annotations

args_dict = {
    "loader_demo": loader_demo,
    "make_annotations": make_annotations
}

if __name__ == "__main__": handle_args(args_dict, "Data")