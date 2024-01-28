import pickle
import os
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from data import MyDataset
from tqdm import tqdm
from train_unet import get_pesq, get_stoi, calculate_snr
from args import config
args= config()

def test(model, data_path, checkpoint_path, batch_size, transform,  save_dir=None):
    print("Loading the data...")
    dataset = MyDataset(data_path, transform=transform)
    
    # Assuming the entire dataset is for testing
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # load checkpoint
    checkpoint= torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model.to(device)

    loss_function = nn.MSELoss()
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    total_pesq_score=0.0
    total_stoi_score=0.0
    total_snr_score=0.0

    file_names=[]
    spec=[]
    phase=[]
    with torch.no_grad():
        for data in  tqdm(test_loader):
            noisy_spec_test, original_spec_test,  noisy_phase_test, original_singal_test, file_name_test = data
            noisy_spec_test, original_spec_test = noisy_spec_test.to(device), original_spec_test.to(device)
            predicted_test = model(noisy_spec_test)
            loss_test = loss_function(predicted_test, original_spec_test)
            test_loss += loss_test.item()

            #Compute PESQ and STOI 
            resize_spec= torchvision.transforms.Resize((args.height, args.width))
            predicted_test= resize_spec(predicted_test)
            pesq_score = get_pesq(original_singal_test, predicted_test, noisy_phase_test)
            stoi_score = get_stoi(original_singal_test, predicted_test, noisy_phase_test)
            snr_score = calculate_snr(original_singal_test.cpu().numpy(), predicted_test.cpu().numpy())
            total_snr_score += snr_score
            total_pesq_score += pesq_score
            total_stoi_score += stoi_score

            #to save
            file_names.append(file_name_test)
            spec.append(predicted_test)
            phase.append(noisy_phase_test)


            # Create result dictionary after the loop
            result_dict = {
                'file_name': file_names,
                'predicted_spec': spec,
                'noisy_phase': phase
            }

            # Save the results outside the loop
            if save_dir:
                save_path = os.path.join(save_dir, 'result_dict.pickle')
                with open(save_path, 'wb') as f:
                    pickle.dump(result_dict, f)

        #Calculating average metrics for the epoch
        average_test_loss = test_loss / len(test_loader)
        average_pesq_score = total_pesq_score / len(test_loader)
        average_stoi_score = total_stoi_score / len(test_loader)
        average_snr_score = total_snr_score / len(test_loader)
        print(f"Test Loss: {average_test_loss:.4f}, Average SNR: {average_snr_score:.4f}, Average PESQ: {average_pesq_score:.4f}, Average STOI: {average_stoi_score:.4f}")

    return average_test_loss, average_snr_score,average_pesq_score, average_stoi_score


