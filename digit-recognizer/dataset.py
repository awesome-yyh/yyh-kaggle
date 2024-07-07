import torch
from torch.utils.data import Dataset
import pandas as pd


class MnistDataSet(Dataset):
    def __init__(self, filename, train=True):
        df = pd.read_csv(filename, skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
        self.images, self.labels = [], []
        for index, row in df.iterrows():
            self.images.append(row.iloc[1:].values.reshape(1, 28, 28))
            self.labels.append(row.iloc[0])
        
        print("image0: ", self.images[0])
        print("label0: ", self.labels[0])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_tensor = torch.tensor(self.images[index], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.int64)
        feature = dict()
        feature["feature"] = image_tensor
        feature["label"] = label_tensor
        
        return feature


if __name__ == "__main__":
    digit_dataset = MnistDataSet(filename = "./train.csv")
    print(len(digit_dataset))
    print(digit_dataset[0])
    
