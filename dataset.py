# define dataset to use pytorch dataset loader

import torch 
import os
import numpy as np
import pandas as pd
from torch.utils .data import Dataset, DataLoader
from torchvision import transforms, utils


class LandmarksDataset(Dataset):
    # root_dir = dataset/train_FAN
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples_indx = os.listdir(root_dir)
    
    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.root_dir,self.samples_indx[idx],"stack_landmarks.csv")
        stack_landmarks = np.loadtxt(input_name, delimiter=',')
        stack_landmarks = stack_landmarks.astype('float').reshape(68,9)#68*9
        target_name = os.path.join(self.root_dir,self.samples_indx[idx],"label_landmarks.csv")
        label_landmarks = np.loadtxt(target_name, delimiter=',')
        label_landmarks = label_landmarks.astype('float').reshape(106*3)
        sample = (stack_landmarks,label_landmarks)

        return sample

def main():
    landmarks_dataset = LandmarksDataset(root_dir="dataset/train_FAN")
    for i in range(len(landmarks_dataset )):
        sample = landmarks_dataset[i]
        print(i,sample['input'].shape,sample['label'].shape)
        if i==3:
            break

if __name__=="__main__":
    main()