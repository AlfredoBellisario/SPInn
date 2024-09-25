#Script to download PDB files
import sys, os
import numpy as np

tuple_data = []
data_search = []

with open('/data/dataset.csv', 'r') as f:
    for i in f:
        tuple_data.append(tuple(i.strip().split(',')))
    print(len(tuple_data[0]))
        
    for j in range(len(tuple_data[0])):
        data_search.append(tuple_data[0][j].replace(' ', ''))

fn_tuple = []
fn_random = tuple(np.random.choice(data_search, 10000, replace=False)) #Number of files downloaded

import ftplib
    
savedir = './'
os.chdir(savedir)

ftp = ftplib.FTP('ftp.wwpdb.org')
ftp.login()

for i in range(len(fn_random)):
    try:
        ftp.cwd('/pub/pdb/data/structures/all/pdb/')
        fn_pdb = 'pdb'
        fn_ext = '.ent.gz' 
        fn = fn_pdb + fn_random[i].lower() + fn_ext
        print(fn)
        file = open(fn, 'wb')
        ftp.retrbinary('RETR ' + fn, file.write) 
        file.close()
    
    #Handle file extenstion .cif

    except(FileNotFoundError, IOError):

        ftp.cwd('/pub/pdb/data/structures/all/pdb/')
        fn_pdb = 'pdb'
        fn_ext = '.ent.gz' 
        fn = fn_pdb + fn_random[i].lower() + fn_ext
        print('File: {} does not exist. Downloading file: {}.cif.gz'.format(fn, fn_random[i].lower()))
        
        ftp.cwd('/pub/pdb/data/structures/all/mmCIF/')
        fn_pdb = ''
        fn_ext = '.cif.gz'
        fn = fn_pdb + fn_random[i].lower() + fn_ext
        print(fn)
        file = open(fn, 'wb')
        ftp.retrbinary('RETR ' + fn, file.write)
        file.close()
        
    except:
        print('No .ent or .cif file, skip.')
