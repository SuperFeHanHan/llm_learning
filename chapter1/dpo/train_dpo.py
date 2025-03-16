# 实现dpo
# 参考: https://github.com/wyf3/llm_related/blob/main/train_llm_from_scratch/dpo_train.py
# 核心1: 数据处理，DPODataCollator，DPODataset
# 核心2: 基于Trainer实现DPOTrainer

import torch, json, os
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['ACCELERATE_TORCH_DEVICE'] = device # 不让accerlerate移动到mps

class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, features):
        inputs_ids = []  # features: list of list, len(features)=bs=16， len(features[0])=3，里面的元素就是DPODataset返回的元素, 前面放chosen的，后面放rejected的。长度为32。
        labels = []

        for feature in features: # chosen
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0] * len(feature[0]) + feature[1])
        for feature in features: # rejected
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0] * len(feature[0]) + feature[2])

        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids] # 做截断处理
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []

            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len: # 做padding
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
                    label = label + [self.tokenizer.pad_token_id] * (max_len - len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels

        inputs_ids, labels = process(inputs_ids, labels)  # 截断 + padding

        return {
            "input_ids": torch.tensor(inputs_ids).to(device),
            "labels": torch.tensor(labels).to(device)
        }


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __getitem__(self, index):
        sample = self.datas[index]  # json读取进来的list of dict
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n, generation_prompt是'<|im_start|>assistant\n'
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]
        return [prompt_inputs, chosen_inputs, rejected_inputs]

    def __len__(self):
        return len(self.datas)


def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))

    return new_logits


def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)

    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)  # (batch_size * 2, seq_len)
    chosen_probs, reject_probs = split_probs(probs)  # (batch_size, seq_len), (batch_size, seq_len)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


class MyDPOTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels=labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)  # (batch_size * 2, seq_len)， 这部分把label=0的输入部份也带上了
        ref_probs = mask_logits(ref_probs, labels)  # 去除输入部份之后的logits之和
        logits = model(input_ids=input_ids, labels=labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss


if __name__ == "__main__":
    print(device)

    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    CUR_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'dpo')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'dpo-output')

    data_path = os.path.join(DATA_DIR, 'dpo_1000.json')  # 数据路径

    args = TrainingArguments(output_dir=OUTPUT_DIR,
                             num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                             do_train=True,
                             per_device_train_batch_size=16,
                             gradient_accumulation_steps=4,
                             # max_steps=15000,
                             logging_steps=50,
                             # report_to='tensorboard',  # tensorboardX
                             save_total_limit=3,
                             bf16=True,
                             learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                             lr_scheduler_type='cosine',
                             dataloader_num_workers=1,
                             dataloader_pin_memory=True,
                             save_safetensors=False,
                             save_steps=100,
                             use_cpu=True  # 是否放置到cpu上
                             )
    print(args.device)
    # args.device = device
    data_collator = DPODataCollator(tokenizer, max_seq_len=512)  # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    dataset = DPODataset(data_path, tokenizer=tokenizer)
    trainer = MyDPOTrainer(model=model, args=args,
                           train_dataset=dataset, tokenizer=tokenizer,
                           data_collator=data_collator
                           )

    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/dpo-1-epoch')
    trainer.save_state()