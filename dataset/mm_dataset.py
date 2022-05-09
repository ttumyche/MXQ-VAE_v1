import os
import re
import csv
import json
import copy
import pickle
import random
from glob import glob
from PIL import Image
from io import BytesIO
from urllib import request

import torch
from torchvision import datasets

from torch.utils.data import Dataset
from transformers import BertTokenizer

class MNIST_Dataset(Dataset):
    def __init__(self, args, base_folder, transform, train=True, test_dset_same=True):
        self.args = args
        self.train = train
        self.transform = transform

        self.base_folder = base_folder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if train:
            self.img_data = glob(self.base_folder + '/mnist_train_img/*.png')
            self.txt_data = glob(self.base_folder + '/mnist_train_text/*.txt')
        else:
            self.img_data = glob(self.base_folder + '/mnist_test_same_img/*.png')
            self.txt_data = glob(self.base_folder + '/mnist_test_same_text/*.txt')

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_filename = self.img_data[idx]
        txt_filename = self.txt_data[idx]

        ori_img = Image.open(img_filename).convert('RGB')
        image, trans_ori_img = self.transform(ori_img)

        caption = open(txt_filename, 'r').readline()

        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                padding='max_length', truncation=True, max_length=self.args.token_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()
        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption

class Flower_Dataset(Dataset):
    def __init__(self, args, transform, train=True):
        self.args = args
        self.train = train
        self.transform = transform

        self.img_path = 'path/to/oxfordflower102/jpg'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if train:
            self.data = [json.loads(l) for l in open('path/to/train/dataset')]
        else:
            self.data = [json.loads(l) for l in open('path/to/valid/dataset')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data[idx]['img_id']
        caption = self.data[idx]['caption']
        label = self.data[idx]['label']

        ori_img = Image.open(os.path.join(self.img_path, img_filename)).convert('RGB')
        image, trans_ori_img = self.transform(ori_img)

        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                padding='max_length', truncation=True, max_length=self.args.token_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()

        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption


