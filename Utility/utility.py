import sys

def handle_args(args_dict, module_name):
    if len(sys.argv) != 2 or sys.argv[1] not in args_dict.keys(): 
        print("Usage: ")
        for key in args_dict.keys(): 
            print(f"python3 -m {module_name} {key}")
    else: 
        args_dict[sys.argv[1]]()