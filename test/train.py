#!/home/anaconda3/envs/MLenv/bin/python
#SBATCH --gres=gpu:1
#SBATCH --job-name=asupp
#SBATCH --mem=24GB
#SBATCH --partition=fast
#SBATCH --ntasks=1 
#SBATCH -C RTX2080Ti 
#SBATCH --output=asupp.test
#SBATCH --ntasks=1

################################################

import tensorflow as tf
import numpy as np
import h5py as h5
#from tensorflow.keras import backend as K
from model.py import spi_nn as model

################################################

# Data Setup
training_data_size = 9900
test_data_size = 100
dim = 128

total = training_data_size + test_data_size
file_h5 = "/dataset.h5"

with h5.File(file_h5, 'r') as data:
    protein_structure = data['real_space'][:training_data_size, :, :]
    fourier_space = data['data_fourier'][:training_data_size, :, :]

# Initialization of arrays
array_intensities = np.zeros((training_data_size, dim, dim))
array_support = np.zeros((training_data_size, dim, dim))
array_protein = np.zeros((training_data_size, dim, dim))
array_autocorrelation = np.zeros((training_data_size, dim, dim))
autocorrelation_float = np.zeros((training_data_size, dim, dim))

# FFT and reshaping operations
for i in range(training_data_size):
    array_intensities[i] = np.abs(np.fft.fftshift(np.fft.fftn(fourier_space[i]))) ** 2
    array_protein[i] = np.abs(protein_structure[i])
    array_support[i] = np.where(array_protein[i] > 0, 1, array_protein[i])
    array_autocorrelation[i] = (np.abs(np.fft.fftshift(np.fft.ifftn(array_intensities[i]))) > 1e-15)
    autocorrelation_float[i] = np.abs(np.fft.fftshift(np.fft.ifftn(array_intensities[i])))

# Normalization and reshaping
for i in range(training_data_size):
    array_intensities[i] = array_intensities[i] / array_intensities[i].max()
    array_protein[i] = array_protein[i] / array_protein[i].max()

# Adding a channel dimension
reshaped_intensities = reshaped_intensities.reshape(reshaped_intensities.shape + (1,))
reshaped_protein = reshaped_protein.reshape(reshaped_protein.shape + (1,))

# Load model (imported as `model`)
spi_model = model()

# Compile model
spi_model.compile(optimizer='adam', loss='mse', lr=0.01)

# Train the model
spi_model.fit(x=reshaped_intensities[:], y=reshaped_protein[:], epochs=50, shuffle=True, batch_size=64)

# Save model weights
spi_model.save_weights("model_weights.h5")
