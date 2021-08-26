import h5py as h
import numpy as np
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--size',type=int,default=35,
                   help="Size of resulting cmaps.")
parser.add_argument("--n_res",type=int,default=None,
                    help="number of residues [(n_res/size) must be int]")
parser.add_argument('--ca_cutoff',type=int,default="8",
                    help="Contact distance in Angstroms")
parser.add_argument('--cutoff',type=int,default="0",
                    help="Avg value for a positive contact")

def pool_arr(arr,pool_size=25,cutoff=0.0,preapply=0.8):             
    shape = arr.shape
    if preapply:
        arr = (arr < preapply) * 1
    try:                    
        assert shape[0] == shape[1]          
        shape = shape[0]                  
    except AssertionError:
        print("Only works for square arrays")
        raise NotImplementedError
    if shape % pool_size != 0:
        print("Please change pool_sizen"
              "None Integer division error")
        raise ArithmeticError
    iters = int(shape/pool_size)
    pooled = np.empty((iters,iters))
    for i in range(iters):
        ilow = int(i*pool_size)
        ihgh = int((i+1)*pool_size)
        for j in range(iters):
            jlow = int(j*pool_size)
            jhgh = int((j+1)*pool_size)
            cur_arr = arr[ilow:ihgh,jlow:jhgh]
            pooled[i,j] = 1 if np.average(cur_arr) > cutoff else 0
    return pooled

if __name__ == '__main__':
    args =  parser.parse_args()
    size =  args.size
    n_res = args.n_res
    ca_cutoff = args.ca_cutoff / 10 # Convert to nanometres from angstroms
    cutoff= args.cutoff

    hdf_path = "./cov_cmaps_hdf.h5"
    with h.File(hdf_path,"a") as hdf:
        p_group = hdf.require_group("pooled_cmaps")
        all_data = hdf["virts/all_data"]
        n_frames,n_residues,_ = all_data.shape
        print(n_residues)
        if n_res:
            n_residues = n_res
        if n_residues % size != 0:
            remain = n_residues % size
        else:
            remain = 0
        n_residues = n_residues - remain
        pool_size  =  n_residues / size

        if f"{size}x{size}_cmaps" in p_group.keys():
            print("Dataset already exists!")
            if input("Delete?").lower() in  ["y","yes"]:
                del p_group[f"{size}x{size}_cmaps"]

        new_dset = p_group.create_dataset(f"{size}x{size}_cmaps",
                               shape=(n_frames,size,size), 
                               dtype=np.int8)
        step = int(n_frames / 100) # Increase this if less RAM available
        slices = [slice(i*step,(i+1)*step) for i in range(100)]
        for s in tqdm(slices):
            data = all_data[s]
            data = data[:, :n_residues,
                           :n_residues]
            data = [pool_arr(x,pool_size=pool_size,
                         cutoff=cutoff,
                         preapply=ca_cutoff) for x in data]
            data = np.stack(data)
            new_dset[s] = data


