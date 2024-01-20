import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from data import MyDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_unet import get_pesq, get_stoi

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
    total_pesq_score=0.0
    total_stoi_score=0.0
    with torch.no_grad():
        for data in  test_loader:
            noisy_spec_test, original_spec_test,  noisy_phase_test, original_singal_test = data
            noisy_spec_test, original_spec_test = noisy_spec_test.to(device), original_spec_test.to(device)
            predicted_test = model(noisy_spec_test)
            loss_test = loss_function(predicted_test, original_spec_test)
            test_loss += loss_test.item()

            #Compute PESQ and STOI 
            pesq_score = get_pesq(original_singal_test, predicted_test, noisy_phase_test)
            stoi_score = get_stoi(original_singal_test, predicted_test, noisy_phase_test)
            total_pesq_score += pesq_score
            total_stoi_score += stoi_score

        #Calculating average metrics for the epoch
        average_test_loss = test_loss / len(test_loader)
        average_pesq_score = total_pesq_score / len(test_loader)
        average_stoi_score = total_stoi_score / len(test_loader)
        print(f"Test Loss: {average_test_loss:.4f}, Average PESQ: {average_pesq_score:.4f}, Average STOI: {average_stoi_score:.4f}")

    return average_test_loss, average_pesq_score, average_stoi_score
