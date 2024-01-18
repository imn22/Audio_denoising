import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data import MyDataset
import matplotlib.pyplot as plt

def train(model, data_path, batch_size, n_epochs, transform, save_path=None):
    print("Loading the data...")
    dataset= MyDataset(data_path, transform=transform)
    train_set, val_set= random_split(dataset, [0.81, 0.19])
    # data loader
    train_loader= DataLoader(train_set, batch_size= batch_size, shuffle= True)
    val_loader= DataLoader(val_set, batch_size= batch_size, shuffle= True)

    device=("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    loss_function = nn.MSELoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  
    train_losses = []
    val_losses = []

    # Training loop
    print('\n start training')
    for epoch in range(n_epochs):
        model.train()  
        train_loss = 0.0

        for noisy, original in train_loader:
            noisy, original = noisy.to(device), original.to(device)
            predicted = model(noisy)
            loss = loss_function(predicted, original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        average_train_loss = train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation
        model.eval()  
        val_loss = 0.0

        with torch.no_grad():
            for noisy_val, original_val in val_loader:
                noisy_val, original_val = noisy_val.to(device), original_val.to(device)
                predicted_val = model(noisy_val)
                loss_val = loss_function(predicted_val, original_val)
                val_loss += loss_val.item()
        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

    # Plot training and validation losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()