import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data import MyDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def test(model, data_path, batch_size, transform,  save_path=None):
    print("Loading the data...")
    dataset = MyDataset(data_path, transform=transform)
    
    # Assuming the entire dataset is for testing
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model.to(device)

    loss_function = nn.MSELoss()
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for noisy_val, original_val in test_loader:
            noisy_val, original_val = noisy_val.to(device), original_val.to(device)
            predicted_val = model(noisy_val)
            loss_val = loss_function(predicted_val, original_val)
            test_loss += loss_val.item()

    average_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {average_test_loss:.4f}")


    #ajouter les metrics mentionnées dans l'énnoncé
    return average_test_loss
