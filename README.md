# Audio Denoising

## Project Description

This project aims to develop a model for audio denoising, specifically focusing on removing background noise from voice recordings. The dataset provided includes:

- A folder containing clean voice recordings (audio/voice_origin/train)
- A folder containing voice recordings with street ambiance in the background (audio/denoising/train)

The correspondence between a recording with ambiance and the perfect voice recording is maintained through file names.

For testing purposes, there are two similar sets of files available in audio/voice_origin and audio/denoising. Additionally, there is a small-sized dataset named `train_small` in both directories, which can be used for quick experimentation.

## Objectives

The main objective is to estimate the clean voice signal from the noisy signal. The signals have a Signal to Noise Ratio (SNR) ranging from 0 to 20 dB.

You have the choice to work on:

- Spectrograms, for example, by employing masking approaches presented in course 09 and estimating masks using a Seq2Seq network or a UNet.
- Waveform directly, with reference to research such as Wave-U-Net or Time-domain Audio Separation Network (TaSNet).

You are free to choose the loss function suitable for the input data format and the neural network architecture.

## Evaluation

In addition to the loss function, the performance evaluation will focus on the Perceptual Evaluation of Speech Quality (PESQ) and the Short-Time Objective Intelligibility (STOI) of the estimated voices.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Audio_denoising.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the dataset provided in the `audio/voice_origin` and `audio/denoising` directories.

4. Experiment with different neural network architectures and loss functions based on your preferences and the project objectives.

5. Evaluate the model's performance using PESQ and STOI metrics on the test dataset.

## References

- A. Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Network", ISMIR 2017.
- D. Stoller et al., "Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation", ISMIR 2018.
- Y. Luo et al., "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech Separation", ICASSP 2018.
- Y. Luo et al., "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation", IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.

