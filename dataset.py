import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class Pheme_Dataset(Dataset):
    def __init__(self, tokenizer, csv_path, image_folder, image_size, max_text_len, id_file, training=False):
        # training: 训练为True，验证为False
        self.csv = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.training = training
        self.max_text_len = max_text_len
        with open(id_file) as file:
            self.image_ids = file.readlines()
        self.image_ids = [int(_id.strip()) for _id in self.image_ids]
        self.transform = get_transforms(image_size)
        self.tokenizer = tokenizer
        self.imgid2sen = dict()
        self.imgid2label = dict()
        for _, row in self.csv.iterrows():
            self.imgid2sen[int(row['image_id'])] = row['text'].split('http')[0].strip() # 清洗掉后面的网址
            self.imgid2label[int(row['image_id'])] = row['label']

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_folder,str(image_id)+'.jpg')) # 读取图像
        image = self.transform(image)
        
        sentence = self.imgid2sen[image_id]
        res = self.tokenizer.encode_plus(text=sentence, truncation=True, padding='max_length', max_length=self.max_text_len, return_tensors='pt', return_attention_mask=True)
        input_ids = res['input_ids']
        attention_mask = res['attention_mask']
        
        label = torch.zeros((1))
        label[0] = self.imgid2label[image_id]
        return image, input_ids.squeeze(0), attention_mask.squeeze(0), label.long()

def get_transforms(image_size):
    # 图像预处理，resize，裁剪，转换为tensor，normalize
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
    return transform