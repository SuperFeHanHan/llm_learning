import os
from transformers import TrainingArguments, Trainer
from model import SiglipModel, SiglipConfig
from dataset import SiglipDataset, MyDataCollator
from transformers import AutoTokenizer, AutoProcessor
# from transformers import ViTImageProcessor, ViTForImageClassification

from datetime import datetime

def train(text_model_name_or_path,
          vision_model_name_or_path,
          text_data_path, image_data_path):
    
    config = SiglipConfig(vision_model_name_or_path,
                          text_model_name_or_path)
    
    model = SiglipModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
  
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = TrainingArguments(
        output_dir=f'./outputs_{cur_time}',
        do_train=True,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    dataset = SiglipDataset(text_data_path=text_data_path,
                            image_data_path=image_data_path,
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=MyDataCollator()
    )
    trainer.train()  # resume_from_checkpoint=True
    trainer.save_model()
    trainer.save_state()
    
if __name__ == '__main__':
    text_model_name_or_path='hfl/chinese-roberta-wwm-ext'
    vision_model_name_or_path='google/vit-base-patch16-224'
    
    data_dir = os.path.join(os.getcwd(), 'data', 'Flickr30k-CN')
    
    image_data_path = os.path.join(data_dir, 'train_imgs.tsv')
    text_data_path = os.path.join(data_dir, 'train_texts.jsonl')

    # image_data_path = os.path.join(data_dir, 'valid_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'valid_texts.jsonl')
    
    # image_data_path = os.path.join(data_dir, 'test_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'test_texts.jsonl')

    train(text_model_name_or_path,
          vision_model_name_or_path,
          text_data_path, image_data_path)