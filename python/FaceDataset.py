import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
import utils
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, mode, indices_file_path, img_path, labels_path, transform=None, transform2=None, target_transform=None):
        '''
            @param mode : "train" | "test"
            @param train_proportion : percentage of data in train dataset.
        '''

        self.img_labels_dir = labels_path
        self.img_dir = img_path
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.indices_file_path = indices_file_path

        # Loading indices
        self.indices = self.load_indices(indices_file_path)
        print(f"FaceDataset: {len(self.indices)} indices loaded.")

        # Loading labels
        self.img_labels = self.load_labels(self.indices)
        print(f"FaceDataset: {len(self.indices)} labels loaded.")
        
        # Extract dataset size
        SIZE = len(self.img_labels)


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        id = str(idx).zfill(6)
        filename = f"c_256_{id}.png"
        img_path = os.path.join(self.img_dir, filename)
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.transform2:
            image = self.transform2(image)

        if self.target_transform:
            label = self.target_transform(label)

        label = label.flatten()
        return image, label
    
    def read_landmarks(self, file_path):
        result = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.split(' ')
                if(len(tokens) < 2):
                    # print(f"WARN: (read_landmarks) in line:'{line}'\n {str(tokens)} \n Found less than 2 tokens. filepath={file_path}")
                    continue
                x = float(tokens[0])
                y = float(tokens[1])
                result.append({"x":x,"y":y})
        if(len(result) != 70):
            raise Exception(f"Feature length must be 70. Currently: {len(result)} {file_path}")
        return result
    

    def landmarks_to_np_array(self, landmarks):
        np_arr = np.zeros((len(landmarks), 2)) # (70, 2)
        i = 0
        for landmark in landmarks:
            np_arr[i, 0] = landmark["x"]
            np_arr[i, 1] = landmark["y"]
            i += 1
        return np_arr


    def load_labels(self, indices):
        landmarks_tab = []
        for index in indices:
            id = str(index).zfill(6)
            filename = f"c_256_ldkms_{id}.txt"
            path = os.path.join(self.img_labels_dir, filename)
            landmarks = self.read_landmarks(path)
            landmarks = [landmarks[30], landmarks[68], landmarks[69], landmarks[48], landmarks[54]]
            landmarks_tab.append(self.landmarks_to_np_array(landmarks))
        return landmarks_tab



    def load_indices(self, path):
        tmp_tab = []
        file = open(path, "r")
        lines = file.readlines()
        for line in lines:
            if len(line) == 0 : 
                continue
            tmp_tab.append(int(line))
        file.close()
        return tmp_tab
