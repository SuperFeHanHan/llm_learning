训练模型的流水线：
1. 定义对应的问题
2. 找对应的数据集: `dataset.py`
```python
from torch.utils.data import Dataset
class SiglipDataset(Dataset):
    def __init__(self, ):
        super().__init__()
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

class MyDataCollator:
    """主要是把数据集拼成batch"""
    def __init__(self):
        pass
    
    def __call__(self, batch):
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }
```
3. 写对应的模型:
    - `MyOutput(ModelOutput)`
    - `MyConfig(PretrainedConfig)`
    - `MyModel(PretrainedModel)`
```python
import torch
from dataclasses import dataclass
from transformers.utils import ModelOuput
@dataclass
class SiglipOutput(ModelOuput):
    loss: torch.FloatTensor = None
    # ...


from transformers import PretrainedConfig
class SiglipConfig(PretrainedConfig):
    model_type = "siglip"  # 为了便于导入
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 一般定义下模型组成部分的路径
        self.vision_model_name_or_path = vision_model_name_or_path

from transformers import PretrainedModel
class SiglipModel(PretrainedModel):
    config_class = SiglipConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.process = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
        pass

    def forward(self, input_ids, attention_mask, pixel_values):
        # 输入由data_collator返回的key对应

        # 默认根据返回的loss进行优化
        return SiglipOutput(loss = ...)
```
4. 写训练脚本
```python
def train():
    config = SiglipConfig(vision_model_name_or_path='/home/user/wyf/train_siglip_from_scratch/vit-base-patch16-224',
                          text_model_name_or_path='/home/user/wyf/chinese-roberta-wwm-ext')
    
    model = SiglipModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
  
    args = TrainingArguments(
        output_dir='./outputs',
        do_train=True,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='none',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    dataset = SiglipDataset(text_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_texts.jsonl',
                            image_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_imgs.tsv',
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=MyDataCollator(tokenizer)
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    trainer.save_state()
    
if __name__ == '__main__':
    train()
```