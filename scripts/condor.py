#!/home/albelli/anaconda3/envs/Condor/bin/python
#SBATCH --gres=gpu:1
#SBATCH --job-name=pdb
#SBATCH --array=6
#SBATCH --partition=fast
#SBATCH --ntasks=1 
#SBATCH --exclude=a026
#SBATCH -C RTX2080Ti 
#SBATCH --output=more-pdb-%a.test
#SBATCH --ntasks=1

import spimage
import spsim
import sys
print(sys.version_info)
import numpy as np
import condor
import os
import h5py
from eke import conversions, rotmodule, slurm_tools

try:
    index = slurm_tools.get_array_index()
except:
    index = 0
print(index)

photon_energy=8000.   # 1.2 keV
intensity=5e-3     # originally 0.05e-3
detector_distance=0.3 # 20 cm
downsample = 2

source = condor.Source(wavelength=conversions.ev_to_m(photon_energy), # eV
                       pulse_energy=intensity, # J
                       focus_diameter=200e-9) # m, originally 1e-6 
detector = condor.Detector(distance=detector_distance,
                           pixel_size=downsample*200e-6,
                           nx=1024//downsample, ny=1024//downsample, # Detector size in pixels
                           cx=512//downsample, cy=512//downsample) # Detector center in pixels

data_directory = "/home/albelli/morePDBfiles/pdb_files/go_simulations"
output_dir = "/home/albelli/morePDBfiles/pdb_files/simulations/"

files = os.listdir(data_directory)
files_to_sim = []

for i in files:
    if i.endswith('ent'):
        files_to_sim.append(i)

num_patterns = 20

rot_a = rotmodule.random(num_patterns).reshape((num_patterns,4))
int_a = np.zeros((num_patterns,)+detector.get_mask().shape)
data_fourier_a = np.zeros((num_patterns,)+detector.get_mask().shape,dtype=complex)
file_a = np.empty([num_patterns], dtype = 'object')

for i in range(index*num_patterns, (index+1)*num_patterns):
    
    file_name = files_to_sim[i]
    
    pdb_file = os.path.join(data_directory, file_name)
    
    print("Start of calc!")        
    # Random orientation (using the quaternion formalism for rotations)
    j = i % num_patterns
    particle = condor.ParticleAtoms(pdb_filename=pdb_file, rotation_values=rot_a[j], rotation_formalism="quaternion")

    experiment = condor.Experiment(source, {"particle_atoms": particle}, detector)
    result = experiment.propagate()
    
         
    all = result
    file_a[j] = file_name#np.string_(file_name)
    
    int_a[j] = result["entry_1"]["data_1"]["data"] 
    data_fourier_a[j] = result["entry_1"]["data_1"]["data_fourier"]
        
    print('File {} simulated'.format(file_name))
    print('File {} simulated'.format(file_a[j]))
    print(int_a[j].shape)
    print(rot_a[j].shape)
    print('Ok.')
    
fname = output_dir + 'pdb_out-one-{}.h5'.format(index)
with h5py.File(fname,'w') as f:
    f.create_dataset("data_fourier", data = data_fourier_a)
    f.create_dataset("rotation", data = rot_a)
    f.create_dataset("intensity", data = int_a)
    f.create_dataset("name", data = file_a, dtype=h5py.special_dtype(vlen=bytes))
