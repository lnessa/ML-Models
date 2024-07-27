import torch
import pandas as pd
from torch.utils.data import DataLoader

def train(M, loss_fn, model_name, train_dataset, test_dataset, batch_size, debug = False):

    N = len(train_dataset)
 
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cuda_available = torch.cuda.is_available()
    print("CUDA available: ", cuda_available)

    device = torch.device('cuda' if cuda_available else 'cpu')

    model = M().to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000

    # imperatively accumulate the loss of each batch
    train_losses = []
    validation_losses = []
    lowest_loss = 10e10

    for epoch in range(num_epochs):
            
        model.train()

        total_batch_loss = torch.tensor(0, dtype=torch.float32).to(device)

        for img in data_loader:
            loss = loss_fn(model, img, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batch_loss += loss

            if debug: 
                print(loss)
                save_model(model, f"./trained_models/{model_name}.pt")
        
        av_loss = (total_batch_loss * batch_size / N).sqrt().data.item()
        train_losses.append(av_loss)

        model.eval()

        with torch.no_grad():
            _, img =  next(enumerate(DataLoader(test_dataset, batch_size=1)))
            loss = loss_fn(model, img, device)
            validation_losses.append(loss.sqrt().data.item())
        
        if av_loss < lowest_loss: save_model(model, f"./trained_models/{model_name}.pt")

        save_losses(train_losses, validation_losses, f"./losses/{model_name}_loss.csv")

        print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]}, Validation Loss: {validation_losses[-1]}")


def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def save_losses(train_losses, validation_losses, filename):
    losses = torch.tensor([train_losses, validation_losses]).T
    pd.DataFrame(data = losses, columns=["Train Loss", "Validation Loss"]).to_csv(filename)