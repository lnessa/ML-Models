import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Model.UNET import load_unet
from Model.Encoder import load_encoder
from Data.DataLoader import Positions, HeatMaps
from Utility.utility import calc_scores

def unet_demo():
    unet = load_unet("./trained_models/unet1.pt")
    print(sum(p.numel() for p in unet.parameters() if p.requires_grad))

    test_dataset = HeatMaps("test")
    N = len(test_dataset)
    print(N)

    _, axs = plt.subplots(5, 5)

    for i in range(5):
        data, _ = test_dataset[60 + i]
        label = unet(data.unsqueeze(0)).detach()

        axs[i][0].imshow(data[0, :, :].detach())
        axs[i][1].imshow(label[0, 0, :, :], vmin = 0, vmax = 1)
        axs[i][2].imshow(label[0, 1, :, :], vmin = 0, vmax = 1)
        axs[i][3].imshow(label[0, 2, :, :], vmin = 0, vmax = 1)
        axs[i][4].imshow(label[0, 3, :, :], vmin = 0, vmax = 1)

    plt.show()

def encoder_demo():
    encoder = load_encoder("./trained_models/unet_positions.pt")
    print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))

    test_dataset = Positions("test")
    N = len(test_dataset)
    print(N)

    _, all_batch =  next(enumerate(DataLoader(test_dataset, batch_size=N)))
    data, truth = all_batch

    pred = encoder(data).detach()

    truth = torch.argmax(truth, 1)
    pred = torch.argmax(pred, 1)

    for i in range(3):
        precision, recall, accuracy, f_score = calc_scores(i, truth, pred)
        print(precision, recall, accuracy, f_score)




