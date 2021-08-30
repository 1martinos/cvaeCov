import argparse
import pickle
import torch
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='load images.')
parser.add_argument('folder',type=str)
parser.add_argument('epoch',type=str,default="",nargs='?')

def read_pickle(file_name):
    imgs, recons = [], []
    with open(file_name,"rb") as f_open:
        while True:
            try:
                im,re = p.load(file_name)
                imgs.append(im)
                recons.append(re)
            except EOFError:
                break
    imgs = torch.stack((imgs))
    recons = torch.stack((recons))
    return imgs, recons

args = parser.parse_args()
folder = args.folder
if folder[-1] == "/":
    folder = folder[:-1]
epoch = args.epoch
new_dir = f"{folder}/imgs"
if not os.path.isdir(new_dir):
    os.makedirs(new_dir)
print("Searching folder: ",folder+f"/{epoch}*.p")
files = sorted(glob(folder+f"/{epoch}*.p"))
fig,axarr = plt.subplots(2)
cbar = None
cbar_ax = cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
for i,f in enumerate(files):
    epoch = f.rsplit("/",1)[1].replace(".p","")
    with open(f,"rb") as file:
       recons, imgs = pickle.load(file)
    imgs, recons = imgs.cpu(), recons.cpu()
    print(f"File {i} of {len(files)}",f)
    for j,(img,recon) in enumerate(tqdm(zip(imgs,recons))):
        img = axarr[0].imshow(img[0])
        axarr[1].imshow(recon)
        if cbar is None:
            cbar = fig.colorbar(img, cax=cbar_ax)
        fig.savefig(f"{folder}/imgs/epoch-{epoch}-{j}.png")
        axarr[0].clear()
        axarr[1].clear()


