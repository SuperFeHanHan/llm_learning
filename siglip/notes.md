参考：
- https://github.com/wyf3/llm_related/tree/main/train_siglip_from_scratch
- https://github.com/OFA-Sys/Chinese-CLIP

# 模型：
- `google/vit-base-patch16-224`
- `hfl/chinese-roberta-wwm-ext`

# 数据集:
- Flickr30k-CN (2.3G, 148915/5000/5000): https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip
- MUGE 牧歌(2.5G): https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip
- COCO-CN
- M6 Corpus
- Wukong (11G): https://wukong-dataset.github.io/wukong-dataset/benchmark.html


# Siglip损失函数：
- $\frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \sum_{j=1}^{|\mathcal{B}|} \underbrace{\log \frac{1}{1+e^{z_{i j}\left(-t \mathbf{x}_i \cdot \mathbf{y}_j+b\right)}}}_{\mathcal{L}_{i j}}$


