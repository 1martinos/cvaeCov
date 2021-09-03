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
    with h.File(hdf_path,"r") as hdf:
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

        """
        Make sure there is an ~ even split of classes in the data
        """
        APO_frames = []
        ATP_frames = []
        APO_keys = []
        ATP_keys = []
        for key,dset in hdf["data"].items():
            if isinstance(key,bytes): # Some save as bytes? Why?
                key = key.decode("utf-8")
            if "APO" in key:
                APO_frames.append(dset.shape[0])
                APO_keys.append(key)
            elif "ATP" in key:
                ATP_frames.append(dset.shape[0])
                ATP_keys.append(key)
        training_frames = min([sum(APO_frames),sum(ATP_frames)]) # n_frames from each type
        apo_choices = int(training_frames/len(APO_keys))
        atp_choices = int(training_frames/len(ATP_keys)) 
        print(f"Taking {training_frames} / {sum(APO_frames)}\n"
              f"That is {training_frames/len(APO_keys)} per sim\n"
              f"Taking {training_frames} / {sum(ATP_frames)}\n"
              f"That is {training_frames/len(ATP_keys)} per sim\n")
        total_train_size = 2*training_frames
        train_dict = {}
        for key in APO_keys:
            frames = np.arange(len(hdf[f"data/{key}"]))
            train_dict[key] = frames
        leftover = 0 # Count how many more frames we need
        for key in ATP_keys:
            frames = np.arange(len(hdf[f"data/{key}"]))
            if len(frames) < atp_choices:
                choices = frames
                leftover += atp_choices - len(frames)
            else: 
                choices = np.random.choice(frames,size=atp_choices,replace=False)
            train_dict[key] = choices
        print("leftover:",leftover)
        if leftover:
            for key in ATP_keys:
                if len(hdf[f"data/{key}"]) > atp_choices + leftover:
                    print(f"Taking frames from {key}")
                    frames = np.arange(len(hdf[f"data/{key}"]))
                    choices = np.random.choice(frames,size=atp_choices+leftover,
                                               replace=False)
                    train_dict[key] = choices
                    break
        print("ATP_frames:")
        print(sum([len(train_dict[key]) for key in ATP_keys]))
        print(sum([len(train_dict[key]) for key in APO_keys]))
        data_group = hdf["data"]
        start_indx = 0
        train_hdf = h.File(f"./train{size}.h5","w")
        training_dset = train_hdf.create_dataset(f"cmaps",
                               shape=(total_train_size,size,size), 
                               dtype=np.int8)
        labels = []
        indices = []
        for key,choices in train_dict.items():
            print(f"Working on {key}")
            data = data_group[key]
            if n_residues != data.shape[1]:
                data = data[:, :n_residues,
                               :n_residues]
            for (i,index) in enumerate(tqdm(choices)):
                cmap = pool_arr(data[index],pool_size=pool_size,
                                cutoff=cutoff,
                                preapply=ca_cutoff)
                training_dset[start_indx+i] = cmap
                labels.append(key)
                indices.append(index)
            start_indx += len(choices)
        train_hdf.create_dataset("labels",data=labels)
        train_hdf.create_dataset("frame_indices",data=indices)
