from network import *
import torch
from torch import optim
import pickle as pkl
import os
import string
import PDB
import MSA
import numpy as np
import random

NAME="thor_"

BASE="/fslhome/bryceeh/compute/thesis/"
PDB_BASE=BASE+"db/cath/dompdb/"
MSA_BASE=BASE+"data/uniref/a3m/"


MODEL_BASE=BASE+"data/models/" + NAME
LOSS_BASE=BASE+"data/losses/" + NAME

BATCH_SIZE=1
CROP_LENGTH=32
SAVE_FREQ=1000

n=0

val = set()
with open("validation.txt","r") as f:
    for line in f.readlines():
        val.add(line[-1])

# Load model
for model_name in os.listdir(BASE+"data/models"):
    if model_name[:len(NAME)] == NAME:
        m = SAVE_FREQ * int(model_name[len(NAME):])
        if m>n:
            n = m

if n==0:
    print("Creating a new model")
    model = ResNet()
else:
    print("Loading model #", n)
    model = torch.load(MODEL_BASE+str(n//SAVE_FREQ))

if torch.cuda.is_available:
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

#dev = torch.device("cpu")
model.to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
print("Model Parameters: ",model.num_param())

losses = np.zeros((SAVE_FREQ,7))

def train(msa, domain, i, j, k, crop_length=CROP_LENGTH):
    input, target = dataloader(msa, domain, i,j,k, crop_length, dev)
    output = model(input)
    loss = calc_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# This could be improved, saving losses
def calc_loss(output, target):
    total_loss = 0
    for i,key in enumerate(output.keys()):
        loss = criterion(output[key], target[key])
        total_loss += loss
        losses[n%SAVE_FREQ,i] = loss.item()
    return total_loss

pdbs = os.listdir(PDB_BASE)
random.shuffle(pdbs)

for name in pdbs:
    pdb_file = PDB_BASE + name
    pdb = PDB.PDB(pdb_file)
    msa_files = MSA_BASE + name[:2] + '/' + name
    for a,domain in enumerate(pdb.domains):
        msa_file = msa_files + string.ascii_lowercase[a]
        if (name + string.ascii_lowercase[a]) in val: # Check if file in validation
            print(name + string.ascii_lowercase[a], " is in validation set") 
            continue
        print(msa_file.split('/')[-1])
        msa = MSA.MSA(msa_file, full=False)
        if len(msa.seqs) == 1:
            print(name + string.ascii_lowercase[a] + " has no matches :(")
            continue
        N = len(domain)
        [i,j,k] = np.random.randint(0,CROP_LENGTH,3)
        ii = list(range(i,N-1,CROP_LENGTH))
        jj = list(range(j,N-1,CROP_LENGTH))
        kk = list(range(k,N-1,CROP_LENGTH))
        random.shuffle(ii)
        random.shuffle(jj)
        random.shuffle(kk)
        for i,j,k in zip(ii,jj,kk):
            loss = train(msa, domain, i, j, k)

            if n%SAVE_FREQ==0:
                print("Saving model!")
                model_path = MODEL_BASE + str(n//SAVE_FREQ)
                loss_path  = LOSS_BASE + str(n//SAVE_FREQ)
                with open(model_path, 'wb') as f:
                    torch.save(model, f)
                np.save(loss_path, losses)
            n+=1
            print(n,i,j,k,loss)

print("Saving model!")
model_path = MODEL_BASE+"epoch"
loss_path = LOSS_BASE+"epoch"
with open(model_path, 'wb') as f:
    torch.save(model, f)
np.save(loss_path, losses[:n%SAVE_FREQ])
