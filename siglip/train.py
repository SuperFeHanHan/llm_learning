import os
from transformers import TrainingArguments, Trainer
from model import SiglipModel, SiglipConfig
from dataset import SiglipDataset, MyDataCollator
from transformers import AutoTokenizer, AutoProcessor
# from transformers import ViTImageProcessor, ViTForImageClassification

from datetime import datetime

def train(output_dir,
          text_model_name_or_path,
          vision_model_name_or_path,
          data_dir, max_seq_length = 64):
    
    config = SiglipConfig(vision_model_name_or_path,
                          text_model_name_or_path)
    
    model = SiglipModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
  
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=False,
        per_device_train_batch_size=64,  # 64
        learning_rate=1e-4,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=8,  # 8
        # do_eval=True,
        # eval_strategy = "steps",
        # eval_steps = 2000, # 默认和logging_steps一致
        # compute_metrics=compute_metrics,
        logging_strategy = "steps",
        logging_steps = 100,
        # logging_dir=os.path.join(output_dir, 'logs'),
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )

    image_train = os.path.join(data_dir, 'train_imgs.tsv')
    text_train = os.path.join(data_dir, 'train_texts.jsonl')

    # image_dev = os.path.join(data_dir, 'valid_imgs.tsv')
    # text_dev = os.path.join(data_dir, 'valid_texts.jsonl')

    train_dataset = SiglipDataset(text_data_path=text_train,
                                  image_data_path=image_train,
                                  tokenizer=tokenizer,
                                  processor=processor,
                                  max_seq_length=max_seq_length)
    # dev_dataset = SiglipDataset(text_data_path=text_dev,
    #                              image_data_path=image_dev,
    #                              tokenizer=tokenizer,
    #                              processor=processor,
    #                              max_seq_length=max_seq_length)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=dev_dataset,
        data_collator=MyDataCollator()
    )
    trainer.train()  # resume_from_checkpoint=True
    trainer.save_model()
    trainer.save_state()
    
if __name__ == '__main__':
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), 'siglip', 'outputs', f'run_{cur_time}')
    text_model_name_or_path='hfl/chinese-roberta-wwm-ext'
    vision_model_name_or_path='google/vit-base-patch16-224'

    data_dir = os.path.join(os.getcwd(), 'data', 'Flickr30k-CN')
    
    # image_data_path = os.path.join(data_dir, 'train_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'train_texts.jsonl')

    # image_data_path = os.path.join(data_dir, 'valid_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'valid_texts.jsonl')
    
    # image_data_path = os.path.join(data_dir, 'test_imgs.tsv')
    # text_data_path = os.path.join(data_dir, 'test_texts.jsonl')

    train(output_dir, 
          text_model_name_or_path,
          vision_model_name_or_path,
          data_dir)   # text_data_path, image_data_path

    # tensorboard --logdir=siglip/outputs