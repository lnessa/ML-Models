from Utility.utility import handle_args

from .demo import loader_demo, make_human_annotations

args_dict = {
    "loader_demo": loader_demo,
    "make_human_annotations": make_human_annotations
}

if __name__ == "__main__": handle_args(args_dict, "Data")