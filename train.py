import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py as h
from datetime import datetime as dt
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import embed as e
import torch.cuda.amp as amp
torch.set_num_threads(4)
from cvae import cVAE as CVAE
from torch.cuda.amp import autocast 

def vae_loss(recon_x, x, mu, logvar, reduction='mean'):
        BCE = F.binary_cross_entropy_with_logits(recon_x,x,reduction="mean")
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD.sum(1).mean(0, True)
        return BCE, KLD

def get_cur_date():
    cur = str(dt.now())
    cur = cur.replace(" ","_")[2:].rsplit(":",1)[0]
    cur = cur.replace(":","")
    return cur

if __name__ == '__main__':
    cur_date = get_cur_date()
    data_path = "./data/cov_cmaps_hdf.h5"

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Hyper-parameters
    EPOCHS = 50
    l_r = 0.00002
    pool_size = 35
    d_l = 12
    data = h.File(data_path,"r")

    data = data[f"pooled_cmaps/{pool_size}x{pool_size}_cmaps"]
    # Split data into train, validate, test sets at 80/10/10 split
    num_samples, *dsize = data.shape
    print(num_samples)
    n_eval = int(num_samples*0.1)
    choices = np.random.choice([*range(num_samples)],
                               size=n_eval,
                               replace=False)
    choices_eval = np.sort(choices)
    eval_data = torch.tensor(data[choices_eval]).float()
    eval_data = eval_data.view(-1,1,*dsize)
    data = np.delete(data,choices_eval,axis=0)

    choices = np.random.choice([*range(len(data))],
                               size=n_eval,
                               replace=False)
    choices_test = np.sort(choices)
    test_data = torch.tensor(data[choices_test]).float()
    test_data = eval_data.view(-1,1,*dsize)
    data = np.delete(data,choices_test,axis=0)
    # Save test data, test on eval, save to virt dset
    batch_size = int(len(data)/400)
    cvae = CVAE(dsize,d_l)
    gscaler = amp.GradScaler(enabled=True)
    if device.type == "cuda":
        pin = True 
        cvae.to(device)
        print("Running on GPU")
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    else: pin = False
    optim = torch.optim.RMSprop(cvae.parameters(), 
                                lr=l_r,
                                alpha=0.9,eps=1e-08)
    dataLoader = DataLoader(data, batch_size=batch_size,
                            pin_memory=pin,shuffle=True)
    dataLoader_eval = DataLoader(eval_data, batch_size=batch_size,
                            pin_memory=pin,shuffle=True)
    file_dir = f"./models/cov{pool_size}-{cur_date}"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    with open(f"{file_dir}/test_data.p","wb") as f_test:
        pickle.dump(test_data,f_test)
    del test_data
    print("Starting Training...")
    loss_list = []
    for epoch in range(EPOCHS):
        t1 = time.time()
        loss_per_Epoch = 0
        BCE_per_Epoch = 0
        KLD_per_Epoch = 0
        for i, batch in enumerate(tqdm(dataLoader)):
            optim.zero_grad()
            if device.type == "cuda":
                batch = batch.to(device)
            batch = batch.float().view(-1,1,*dsize)
            with autocast():
                reconstruct_x, mean, log_var, z = cvae(batch)
                BCE,KLD = vae_loss(reconstruct_x,
                                 batch.view(-1,*dsize),
                                 mean,
                                 log_var)
                loss = BCE + KLD
            loss_list.append(
                f"EPOCH: {epoch} BATCH: {i} BCE: {BCE.item()}" 
                f"KLD: {KLD.item()} EFF_LOSS: {loss.item()}"
                )

            gscaler.scale(loss).backward()
            gscaler.step(optim)
            gscaler.update()

            loss_per_Epoch += loss.item()
            BCE_per_Epoch += BCE.item()
            KLD_per_Epoch += KLD.item()
            del batch
        t2 = time.time()
        print(f"Time for epoch {epoch}: {round(t2-t1,3)}s")
        loss_eval = 0
        with torch.no_grad():        # Test for each Epoch
            with open(f"./{file_dir}/{epoch}.p","wb") as f:
                for batch_eval in dataLoader_eval:
                    if device.type == "cuda":
                        batch_eval = batch_eval.to(device)
                    reconstruct_x, mean, log_var, z = cvae(batch_eval)
                    BCE,KLD = vae_loss(reconstruct_x,
                                 batch_eval.view(-1,*dsize),
                                 mean,
                                 log_var)
                    loss_eval += BCE + KLD
                    pickle.dump((reconstruct_x.cpu(),
                                 batch_eval.cpu()),
                                 f)
                with open(f"./{file_dir}/test_loss.txt","a") as f:
                    f.write(f"{epoch}    {loss_eval.item()}")
                    f.write("\n")
        del batch_eval, loss_eval
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        print(f"Loss for epoch {epoch}: {round(loss_per_Epoch, 3)}")
        print(f"BCE: {round(BCE_per_Epoch, 3)}  KLD: {round(KLD_per_Epoch, 3)}")
    pickle.dump(cvae, open(f"{file_dir}/pooled{pool_size}-{cur_date}.pickle", "wb"))
    with open(f"./{file_dir}/loss_stats-{cur_date}.txt","w") as f:
        for x in loss_list:
            f.write(x + "\n")
