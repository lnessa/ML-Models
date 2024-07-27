import matplotlib.pyplot as plt

from UNET.UNET import load_unet
from Data.DataLoader import HeatMaps

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
