import os
import numpy as np
import mdtraj as md
import h5py as h
import pickle as p
import json
import parmed as pmd
import argparse
from glob import glob
from numpy.linalg import norm
from itertools import combinations
from tqdm import tqdm
from time import time as t
from matplotlib import pyplot as plt
from IPython import embed as e
parser = argparse.ArgumentParser(description='Simulation directory.')
parser.add_argument('folder',type=str)

class h5select:
    """
    Quick class to check hdf groups recursively.
    Feed it a function using the "select" variable
    and it will evaluate this on all datasets and then 
    only return those chosen
    """
    def __init__(self,group,abspath=True,
                select=None,atomselect=None):
        self.names = []
        self.data = []
        self.group = group
        group.visititems(self)
        select = None
        if abspath:
            root = group.name
            self.names = [os.path.join(root,n) for n in self.names]
        if callable(select):
            select = self.selection(select)
        elif select is True:
            select = self.data
        if callable(atomselect):
            select = [atomselect(ds) for ds in select]
        if select:
            return select

    def selection(self,select):
        if self.data and self.names:
            truth_list = [ds for ds in self.data if select(ds)]
            return truth_list
        else:
            print("No selection made!")

    def as_dict(self):
        return dict(zip(self.names,self.data))

    def __call__(self, name, h5obj):
        if isinstance(h5obj,h.Dataset) and not name in self.names:
            self.names += [name]
            self.data += [h5obj]

    def __len__(self):
        return len(self.names)


def crd_filt(name):
    coords = ["xtc","dcd"]
    if get_extension(name) in coords:
        return True
    else:
        return False

def top_filt(name):
    tops = ["psf","pdb"]
    if get_extension(name) in tops:
        return True
    else:
        return False

def get_extension(name):
    return name.rsplit(".",1)[-1]

def new_strip(traj,n_residues=595): # 3-598 is current range taken 
    for res in traj.top.residues:
        if res.name == "ALA":
            zero_idx = res.index
            break
    res_range = range(zero_idx,zero_idx+n_residues)
    atoms = []
    for i in res_range:
        res = traj.top._residues[i]
        # Convert names from simulation ones, we use 
        # custom parameters for eg. zinc binding
        res.name = map_residues(res.name)
        for atom in res.atoms:
            atoms.append(atom.index)
    return traj.atom_slice(atoms)

def map_residues(res_name):
    # Change residue names that are coordinating
    std_res = ['CYS', 'ASP','SER', 'GLN', 'LYS',
               'ILE', 'PRO','THR', 'PHE', 'ASN', 
               'GLY', 'HIS','LEU', 'ARG', 'TRP', 
               'ALA', 'VAL','GLU', 'TYR', 'MET']
    if res_name in ["CY1","CY2","CY4","CYM"]:
        res_name = "CYS"
    elif res_name in ["HD2","HE1","HEK"]:
        res_name = "HIS"
    elif res_name not in std_res:
        print(f"Strange residue: {res_name}")
        return None
    return res_name

def process_file_name(name):
    splat = name.split("/")
    sim_type = splat[-3][:3]
    f_name = splat[-1].split(".")[0]
    return f"{sim_type}_{f_name}" 

def findfiles(directory):
    fpaths = []
    for root,folder,files in os.walk(directory):
        if files:
            [fpaths.append(os.path.join(root,f)) for f in files]
    fpaths = sorted(fpaths)
    crds = [*filter(crd_filt,fpaths)]
    tops = [*filter(top_filt,fpaths)]
    return tops,crds

def parmed_parse(folder_path):
    """
    gro files throw up some errors in MDTraj with the residue counts,etc.
    so lets use ParmEd to convert them to PDBs and conform the formatting
    on existing PDBs.
    """
    files = glob("./*/*/*.gro")+glob("./*/*/*.pdb")
    for file in files:
        new_name = file.replace("gro","pdb")
        gro = pmd.load_file(file)
        gro.save(new_name,overwrite=True)
        print(f"Converted {file} ---> {new_name}.")
        

def create_virt_dataset(hdf,name,sims):
    """
    This could be better made, simply makes a virtual dataset from a group
    """
    def iter_over_axis(arr,lengths,axis=0):
        iterator = [slice(None)]*len(arr.shape)
        prev = 0
        cur = 0
        for dist in lengths:
            cur += dist
            iterator[axis] = slice(prev,cur)
            prev = cur
            yield tuple(iterator)

    group = hdf.require_group("virts")
    sizes = []
    sources = []
    sele = [dataset for dataset in h5select(sims).data]
    for dataset in sele:
        src = h.VirtualSource(dataset)
        sources.append(src)
        sizes.append(src.shape)
    size = np.array(sizes)
    n_frames = sum(size[:,0])
    max_atoms = max(size[:,1]) 
    max_dim = max(size[:,2])
    vshape = (n_frames,max_atoms,max_dim)
    vtype = dataset.dtype   # take last dataset dtype
    layout = h.VirtualLayout(vshape,vtype)
    iterate = iter_over_axis(layout,size[:,0])
    for source,slices in zip(sources,iterate):
        layout[slices] = source
    result = group.create_virtual_dataset(name,layout)
    return result


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.folder
    parmed_parse(data_path)
    tops,crds = findfiles(data_path)
    data_files = [*zip(tops,crds)]
    print("FILES FOUND:\n")
    for pair in data_files:
        print(pair)
    print("\n")
    hdf_path = "./cov_cmaps_hdf.h5" 
    hdf = h.File(hdf_path,"w")
    data_group = hdf.require_group("data")
    misc = hdf.require_group("misc")
    residue_dict = {}
    trajs = []
    names = []
    labels = []
    for i,(top,crds) in enumerate(data_files):
        # Skip the apo dimers if in dataset
        if "APO" in top and ("MB" in top or "EF" in top):
            continue
        traj = md.load(crds,top=top)
        traj = new_strip(traj)
        name = process_file_name(top)
        name = f"{name}-{i}"
        n_frames = traj.n_frames
        n_residues = traj.n_residues
        step = int(n_frames / 100) # Increase this if less RAM available
        slices = [slice(i*step,(i+1)*step) for i in range(100)]
        cmap_dset = hdf["data"].create_dataset(f"{name}-cmaps",
                                               shape=(n_frames,
                                                      n_residues,
                                                      n_residues),
                                               dtype=np.float32)
        for s in tqdm(slices):
            cmap = md.geometry.squareform(
                *md.compute_contacts(traj[s],scheme="ca")
                )
            cmap_dset[s] = cmap 
        [labels.append(name) for i in range(n_frames)]
    misc.create_dataset("all_labels",data=labels)
    create_virt_dataset(hdf,"all_data",hdf["data"])


