import torch.nn as nn

def unet_loss(unet, data, device = 'cpu'):
    criterion = nn.CrossEntropyLoss().to(device)
    img, label = data
    img = img.to(device)
    label = label.to(device)
    pred = unet(img)
    return criterion(label, pred)

