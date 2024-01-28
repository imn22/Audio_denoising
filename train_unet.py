import os
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from data import MyDataset, retreive_sig
import matplotlib.pyplot as plt
from tqdm import tqdm
from args import config
args= config()


def train(model, data_path, batch_size, n_epochs, transform, save_dir=None):
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
    val_pesq_scores=[]
    val_stoi_scores=[]
    val_snr=[]

    min_loss= 10e7
    # Training loop
    print('\n start training')
    for epoch in tqdm(range(n_epochs)):
        model.train()  
        train_loss = 0.0

        for data in train_loader:
            noisy_spec, original_spec, _, _ , _= data
            noisy_spec, original_spec = noisy_spec.to(device), original_spec.to(device)

            predicted = model(noisy_spec)
            loss = loss_function(predicted, original_spec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        average_train_loss = train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # if train_loss < min_loss:
        #     min_loss= train_loss
        #     save_path = os.path.join(save_dir, f'checkpoint_ep_{epoch}.pth')
        #     print('checkpoint saved at: ', save_path,)
            # torch.save(model.state_dict(), save_path)

        # Validation
        model.eval()
        val_loss = 0.0
        total_pesq_score = 0.0
        total_stoi_score = 0.0
        total_snr_score=0.0

        with torch.no_grad():
            for data in  val_loader:
                noisy_spec_val, original_spec_val,  noisy_phase_val, original_singal_val, _ = data
                noisy_spec_val, original_spec_val = noisy_spec_val.to(device), original_spec_val.to(device)
                predicted_val = model(noisy_spec_val)

                loss_val = loss_function(predicted_val, original_spec_val)
                val_loss += loss_val.item()

                #Compute PESQ and STOI 
                resize_spec= torchvision.transforms.Resize((args.height, args.width))
                predicted_val= resize_spec(predicted_val)
                pesq_score = get_pesq(original_singal_val, predicted_val, noisy_phase_val)
                stoi_score = get_stoi(original_singal_val, predicted_val, noisy_phase_val)
                snr_score = calculate_snr(original_singal_val, predicted_val, noisy_phase_val)
                total_snr_score += snr_score
                total_pesq_score += pesq_score
                total_stoi_score += stoi_score

        #Calculating average metrics for the epoch
        average_val_loss = val_loss / len(val_loader)
        average_pesq_score = total_pesq_score / len(val_loader)
        average_stoi_score = total_stoi_score / len(val_loader)
        average_snr_score = total_snr_score / len(val_loader)
        val_snr.append(average_snr_score)
        val_losses.append(average_val_loss)
        val_pesq_scores.append(average_pesq_score)
        val_stoi_scores.append(average_stoi_score)
        print(f"Epoch {epoch + 1}, Train Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Average PESQ: {average_pesq_score:.4f}, Average STOI: {average_stoi_score:.4f}")

    return train_losses, val_losses,val_snr, val_pesq_scores, val_stoi_scores
    


def get_pesq(original_singal, predicted, noisy_phase):
    #retrive predicted signal
    original_singal= original_singal[1].cpu()
    noisy_phase= noisy_phase.cpu().numpy()
    predicted= predicted.cpu().numpy()
    predicted_sig= retreive_sig(predicted, noisy_phase, args.n_fft, args.hop_length_fft)
    predicted_sig= torch.tensor(predicted_sig)
    # just apply pesq
    pesq = PerceptualEvaluationSpeechQuality(args.fs, 'nb')
    result= pesq(predicted_sig, original_singal)
    return result.item()

def get_stoi(original_singal, predicted, noisy_phase):
    original_singal= original_singal[1].cpu()
    noisy_phase= noisy_phase.cpu().numpy()
    predicted= predicted.cpu().numpy()
    predicted_sig= retreive_sig(predicted, noisy_phase, args.n_fft, args.hop_length_fft)
    predicted_sig= torch.tensor(predicted_sig)

    stoi = ShortTimeObjectiveIntelligibility(args.fs, False)
    result= stoi(predicted_sig, original_singal)
    return result.item()

def calculate_snr(original_signal, predicted, noisy_phase):
    original_signal = original_signal[1].cpu().numpy()
    noisy_phase = noisy_phase.cpu().numpy()
    predicted = predicted.cpu().numpy()
    predicted_sig = retreive_sig(predicted, noisy_phase, args.n_fft, args.hop_length_fft)
    # Directly compute power of the original and noise signals
    signal_power = np.mean(np.square(original_signal))
    noise_power = np.mean(np.square(original_signal - predicted_sig))

    snr = 10 * np.log10(signal_power / noise_power)
    return snr