import os
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
import json

# ===============================
# 1. 加载本地 tokenizer
# ===============================
tokenizer = BertTokenizer.from_pretrained("/home/zhoushengye/PycharmProjects/pythonProject/bert-base-uncased")

# ===============================
# 2. 定义编码函数
# ===============================
def encode_batch(texts, max_length=128):
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

# ===============================
# 3. 处理单个文件
# ===============================
def process_sst2_file(input_path, has_labels=True):
    df = pd.read_csv(input_path, sep='\t')

    if has_labels:
        texts = df['sentence'].tolist()
        labels = df['label'].tolist()
    else:
        texts = df['sentence'].tolist()
        labels = [-1] * len(texts)  # test集无标签，用-1占位

    encodings = encode_batch(texts)

    result = []
    for i in range(len(texts)):
        item = {
            'input_ids': encodings['input_ids'][i],
            'attention_mask': encodings['attention_mask'][i],
            'label': labels[i]
        }
        result.append(item)

    return result

# ===============================
# 4. 执行预处理
# ===============================
root_dir = "/home/zhoushengye/PycharmProjects/pythonProject/raw_glue_data/SST-2"
output_dir = "/home/zhoushengye/PycharmProjects/pythonProject/glue_processed"

os.makedirs(output_dir, exist_ok=True)

print("✅ 正在处理 train.tsv ...")
train_data = process_sst2_file(os.path.join(root_dir, "train.tsv"))
with open(os.path.join(output_dir, "train.json"), 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

print("✅ 正在处理 dev.tsv ...")
dev_data = process_sst2_file(os.path.join(root_dir, "dev.tsv"))
with open(os.path.join(output_dir, "dev.json"), 'w') as f:
    for item in dev_data:
        f.write(json.dumps(item) + "\n")

print("✅ 正在处理 test.tsv ...")
test_data = process_sst2_file(os.path.join(root_dir, "test.tsv"), has_labels=False)
with open(os.path.join(output_dir, "test.json"), 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print("🎉 全部数据已预处理完成，保存在 glue_processed/ 文件夹中！")
