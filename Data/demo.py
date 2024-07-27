import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from .DataLoader import HeatMaps, annotation_dict

def loader_demo():
    test_dataset = HeatMaps("train")
    N = len(test_dataset)
    print(N)

    _, axs = plt.subplots(5, 2)

    for i in range(5):
        data, label = test_dataset[100 + i]
        axs[i][0].imshow(data[0, :, :])
        axs[i][1].imshow(label[3, :, :])

    plt.show()

def make_annotations():
    input_path = "./datasets/glasses/"
    output_path = "./datasets/glasses_annotation/"
    files = os.listdir(input_path)

    for fn in files:
        x1, y1, x2, y2, l = annotation_dict[fn]
        cols = [x + y * 32 for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)]
        df = pd.read_csv(input_path + fn)
        N = df.shape[0]
        df.iloc[:, 1:] = np.zeros((N, 21 * 32))
        for col in cols: df.iloc[:, col + 1] = l
        df.to_csv(output_path + fn, index=False)




