import sys
import torch

def handle_args(args_dict, module_name):
    if len(sys.argv) != 2 or sys.argv[1] not in args_dict.keys(): 
        print("Usage: ")
        for key in args_dict.keys(): 
            print(f"python3 -m {module_name} {key}")
    else: 
        args_dict[sys.argv[1]]()

def calc_scores(label, truth, pred):
    TP = torch.logical_and(truth == label, pred == label).sum()
    FP = torch.logical_and(truth != label, pred == label).sum()
    TN = torch.logical_and(truth != label, pred != label).sum()
    FN = torch.logical_and(truth == label, pred != label).sum()

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    accuracy  = (TP + TN) / (TP + FN + TN + FP)
    f_score   = 2 * precision * recall / (precision + recall)

    return precision, recall, accuracy, f_score