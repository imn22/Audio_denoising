import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def scale_minmax(X, min_val, max_val):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled

def create_spec(data_dir,n_fft, hop_length_fft, noisy=True):
    signal_type = 'noisy' if noisy else 'original'
    print(signal_type)
    spec_dir = os.path.join(data_dir, signal_type , 'spec')
    phase_dir = os.path.join(data_dir, signal_type , 'phase')
    
    # Create directories to store the spectrograms and phases
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)

    # List all files in the data path
    signals_names = os.listdir(os.path.join(data_dir, signal_type , 'signal'))

    for signal_name in signals_names:
        # Load the signal using librosa
        y, sr = librosa.load(os.path.join(data_dir, signal_type , 'signal', signal_name), sr=None)  
        # Compute the STFT
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length_fft)
        # Separate magnitude and phase
        magnitude, phase = librosa.magphase(stft)
        print('shaaape',magnitude.shape)
        # Convert magnitude to decibel scale
        magnitude_db = librosa.amplitude_to_db(np.abs(magnitude), ref=np.max)
        print('max and min magnitude db', np.max(magnitude_db), np.min(magnitude_db))

        # show spectrogram image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(magnitude_db, sr=sr, hop_length=hop_length_fft, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {signal_name}')
        plt.show()
        #save spec
        spec_img_path = os.path.join(spec_dir, signal_name + '.npy')
        np.save(spec_img_path, magnitude_db)

        # Display phase spectrogram
        phase_angle = np.angle(phase)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(phase_angle, sr=sr, hop_length=hop_length_fft, x_axis='time', y_axis='hz', cmap='twilight')
        plt.colorbar(format='%+2.0f rad')
        plt.title(f'Phase Spectrogram of {signal_name}')
        plt.show()
        # Save phase data
        phase_path = os.path.join(phase_dir, signal_name + '.npy')
        np.save(phase_path, phase)

    print(f"Spectrograms and phases stored in '{spec_dir}' and '{phase_dir}' respectively.")

signals_path='data/train_small'

# create_spec(signals_path, 512,256, noisy=True)
def retreive_sig(magnitude_db, phase,n_fft, hop_length_fft ):
    #first go backt othe magnitude scale
    magnitude= librosa.db_to_amplitude(magnitude_db, ref= 1.0)
    #include the phase
    signal_with_phase= magnitude * phase 
    #recsntrct
    audio= librosa.core.istft(signal_with_phase, hop_length=hop_length_fft, n_fft=n_fft)
    return audio