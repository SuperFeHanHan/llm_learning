import torch
import os, json, random
import pandas as pd
from torch.utils.data import Dataset
import base64
from PIL import Image
from io import BytesIO

def load_data(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        # 因为一个文本可能对应多个图片
        for image_id in line['image_ids']:
            res.append({'image_ids': image_id, 
                        'text': line['text']})
    return res


class SiglipDataset(Dataset):
    def __init__(self, text_data_path, 
                 image_data_path,
                 tokenizer, 
                 processor, 
                 max_seq_length=64, 
                 ):
        super().__init__()
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length
        
        # image_ids, text, text_id
        self.datas = load_data(self.text_data_path)
        random.shuffle(self.datas)
        self.images = pd.read_csv(self.image_data_path, 
                                  sep='\t', header=None, 
                                  names=['image_id', 'base64']) 
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        text = sample['text']
        image_ids = sample['image_ids']

        tok = self.tokenizer(text, max_length=self.max_seq_length, 
                             padding='max_length', truncation=True)
        input_ids = tok['input_ids']
        attention_mask = tok['attention_mask']

        image_base64 = self.images.loc[self.images['image_id']==image_ids, "base64"].values[0]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']
    
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        } 

class MyDataCollator:
    # def __init__(self, tokenizer):
    #     self.tokenizer = tokenizer
    
    def __call__(self, features):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        pixel_values = [f['pixel_values'] for f in features]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }

if __name__=="__main__":
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")

    data_dir = os.path.join(os.getcwd(), 'data', 'Flickr30k-CN')
    image_data_path = os.path.join(data_dir, 'valid_imgs.tsv')
    text_data_path = os.path.join(data_dir, 'valid_texts.jsonl')
    
    # image_data_path = os.path.join(data_dir, 'test_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'test_texts.jsonl')
    
    # image_data_path = os.path.join(data_dir, 'train_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'train_texts.jsonl')

    dataset = SiglipDataset(text_data_path=text_data_path,
                            image_data_path=image_data_path,
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)
    
    print(len(dataset))
    print(dataset[2])