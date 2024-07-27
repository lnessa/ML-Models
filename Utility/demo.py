import torch

from VAE.loss import ncut_loss

def ncut_loss_demo():

    latent = torch.zeros(3, 2, 50, 50)
    latent[:, 0, :25, :25] = 0.9
    latent[:, 0, 25:, 25:] = 0.9
    latent[:, 0, 25:, :25] = 0.1
    latent[:, 0, :25, 25:] = 0.1

    latent[:, 1, :25, 25:] = 0.1
    latent[:, 1, 25:, :25] = 0.1
    latent[:, 1, 25:, :25] = 0.9
    latent[:, 1, :25, 25:] = 0.9
    
    img = torch.zeros(3, 2, 50, 50)
    img[0, :, 1:, 1:] = 1
    img[0, :, :1, :1] = 1
    img[1, :, :4, :4] = 1
    img[1, :, 4:, 4:] = 1
    img[2, :, 25:, 25:] = 1
    img[2, :, :25, :25] = 1

    print(ncut_loss(img, latent))
    #print(ncut_loss_(img, latent))

