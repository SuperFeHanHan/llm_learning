
import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
@dataclass
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None


from transformers import PretrainedConfig
class SiglipConfig(PretrainedConfig):
    model_type = "siglip"  # 为了便于导入
    def __init__(
        self,
        vision_model_name_or_path: str = "google/vit-base-patch16-224",
        text_model_name_or_path: str = "hfl/chinese-roberta-wwm-ext",
        **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.text_model_name_or_path = text_model_name_or_path

from transformers import PreTrainedModel 
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
class SiglipModel(PreTrainedModel):
    config_class = SiglipConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.process = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)

        # 损失函数里的参数
        self.t = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        输入由data_collator返回的key对应
        siglip核心思想在这实现
        """
        text_outputs = self.text_model(input_ids, attention_mask)
        vision_outputs = self.vision_model(pixel_values)

        vision_features = vision_outputs[1] # pooler_output
        text_features = text_outputs[1] # pooler_output
        
        # (B, H) -> (B, H)
        vision_features = vision_features / vision_features.norm(p=2, dim=-1, keepdim=True) # l2标准化
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # l2标准化
        
        # (B, H) (H, B) -> (B, B)
        # Step1: t * x_i * y_j + b，原文中说t一般是exp(t')表示
        # 每一行对应的是[文本i-图像1, 文本i-图像2, ...]
        logits_per_text = self.t.exp() * torch.matmul(text_features, vision_features.t()) + self.b
        # 每一行对应的是[图像i-文本1, 图像i-文本2, ...] -> 不参与损失计算，只是为了返回看看
        logits_per_image = logits_per_text.t()
        
        # Step2: z_{ij} * (t * x_i * y_j + b), z_{ij}控制是否相关
        B = logits_per_text.shape[0]
        eye = torch.eye(B, device=logits_per_text.device) # 生成单位矩阵
        # 对角线全为1，非对角线为-1，即成对的图文标签为1，非成对的为-1
        labels = 2*eye - torch.ones_like(logits_per_text, device=logits_per_text.device) 
        
        # Step3: 对Step2进行log(1/(1+exp(-x)))
        loglik = F.logsigmoid(labels * logits_per_text)

        # Step4: 计算每个图片的loglik之和(正确判断正负例的logP之和)，再取负号和batch的均值。
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()

        return SiglipOutput(loss=loss, 
                            logits_per_text=logits_per_text, 
                            logits_per_image=logits_per_image, 
                            text_embeds=text_features, 
                            image_embeds=vision_features)