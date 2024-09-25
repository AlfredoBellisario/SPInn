from chimerax.core.commands import run
from chimerax.map.volume import save_map
import numpy as np
import h5py
import glob
import os

def rebin(x, shape = [128,128]):
    sh = shape[0],x.shape[0]//shape[0],shape[1],x.shape[1]//shape[1]
    return x.reshape(sh).mean(-1).mean(1)

def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')
                  
# The working directory
wd = './'
os.chdir(wd)
run(session, "log text working directory %s" % (wd))

# The number of pixels
N = 256

# The map resolution in Angstrom
r = 3.5

# The axis to project on
axis = -1

# Change this glob to get the PDBs you want to calculate projections on
pdbs = glob.glob('*.ent')

nrfiles = 10900
log_file = "logfile.log"

out_merged = np.zeros((nrfiles,N//2,N//2),dtype=float)
out_fourier_merged = np.zeros((nrfiles,N//2,N//2),dtype=complex)

for index, pdb in enumerate(pdbs):
    run(session, "log text Projecting %s" % (pdb))
    opened_models = run(session, "open %s" % (pdb))
    # You can change the map resolution here. Note that the projections
    opened_map = run(session, "molmap #1 %f" % r)
    proj = np.sum(opened_map.data.matrix(), axis=axis)
    # Pad it to the desired size        
    if (proj.shape[0] > N/2) or (proj.shape[1] > N/2):
        print(pdb)
        try:
            proj_padded = to_pad(proj,(156*2,156*2))
            del proj
            proj = rebin(proj_padded,(156,156))
            print(proj.shape)
        #with open(log_file, "a") as file:  # Open the log file in append mode
        #    file.write(f"Exception occurred for str{pdb}: {int(proj.shape[0])} {int(proj.shape[1])}\n")  # Write the exception to the log file
        except: 
            print('Map is too large for the requested image size')
            ValueError('Map is too large for the requested image size')
    print(pdb,end='\r')
    print(proj.dtype,end='\r')
    print(proj.shape,end='\r')
    print(proj.shape[1],end='\r')
    dimx1 = np.int(np.floor(N//2 - proj.shape[0]/2)) 
    dimx2 = np.int(np.floor(N//2 + proj.shape[0]/2))
    dimy1 = np.int(np.floor(N//2 - proj.shape[1]/2)) 
    dimy2 = np.int(np.floor(N//2 + proj.shape[1]/2))
    out = np.zeros((N,N),dtype=float)
    out_fourier = np.zeros((N//2,N//2),dtype=complex)
    print(dimy1,end='\r')
    print(dimy2,end='\r')
    
    out[dimx1 : dimx2, dimy1 : dimy2] = proj
    rebinned = rebin(out,(128,128))
    out_fourier = np.fft.ifftn(rebinned)
    out_merged[index,:,:] = (rebinned).copy()
    out_fourier_merged[index,:,:] = out_fourier.copy() 

    run(session, "close")
    
fname = 'projections.h5'
with h5py.File(fname,'w') as f:
    f.create_dataset("data_fourier", data = out_fourier_merged, dtype='complex')
    f.create_dataset("intensity", data = out_merged)
