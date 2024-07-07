import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd


class BertDataSet(Dataset):
    def __init__(self, filename, base_model):
        df = pd.read_csv(filename, skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
        
        self.texts = df["text"].tolist()
        self.labels = df["target"].tolist()
        
        print("texts0: ", self.texts[0])
        print("label0: ", self.labels[0])
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.max_length = 160

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text_encode_dict = self.tokenizer(self.texts[index], add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        
        label_tensor = torch.tensor(self.labels[index], dtype=torch.int64)
        
        feature = dict()
        feature['input_ids'] = text_encode_dict['input_ids'][0]
        feature['attention_mask'] = text_encode_dict['attention_mask'][0]
        feature['token_type_ids'] = text_encode_dict['token_type_ids'][0]
        feature["labels"] = label_tensor
        
        return feature


if __name__ == "__main__":
    base_model = "/data/app/yangyahe/base_model/google-bert-bert-base-uncased"
    dataset = BertDataSet(filename = "./nlp-getting-started/train.csv", base_model=base_model)
    print(len(dataset))
    print(dataset[0], dataset[0]["input_ids"].shape)
