import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# ===============================
# 参数配置
# ===============================
TASKS = {
    "SST-2": {"path": "SST-2", "text": "sentence", "num_labels": 2},
    "MRPC":  {"path": "MRPC",  "text": ("sentence1", "sentence2"), "num_labels": 2},
    "RTE":   {"path": "RTE",   "text": ("sentence1", "sentence2"), "num_labels": 2}
}
BASE_DIR = "/home/zhoushengye/PycharmProjects/pythonProject/raw_glue_data"
MODEL_PATH = "/home/zhoushengye/PycharmProjects/pythonProject/bert-base-uncased"
OUTPUT_DIR = "./glue_outputs"

# ===============================
# 固定随机种子
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ===============================
# 评估指标函数
# ===============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# ===============================
# 文本编码函数
# ===============================
def tokenize_function(examples, tokenizer, text_fields):
    if isinstance(text_fields, tuple):
        tokens = tokenizer(
            examples[text_fields[0]],
            examples[text_fields[1]],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    else:
        tokens = tokenizer(
            examples[text_fields],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    unk_id = tokenizer.unk_token_id
    vocab_size = tokenizer.vocab_size
    tokens["input_ids"] = [
        [tok if tok < vocab_size else unk_id for tok in seq]
        for seq in tokens["input_ids"]
    ]

    if "label" not in examples:
        tokens["label"] = [0] * len(tokens["input_ids"])
    else:
        tokens["label"] = examples["label"]

    return tokens

# ===============================
# 主训练流程
# ===============================
for task, config in TASKS.items():
    print(f"\n🚀 正在处理任务：{task}")

    data_path = os.path.join(BASE_DIR, config["path"])
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep="\t")

    # 创建数据集对象
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=config["num_labels"]
    )

    # Tokenize
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, config["text"]), batched=True)
    dev_dataset = dev_dataset.map(lambda x: tokenize_function(x, tokenizer, config["text"]), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer, config["text"]), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 设置训练参数
    task_output_dir = os.path.join(OUTPUT_DIR, task.lower())
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(task_output_dir, "logs"),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        seed=42
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    print(f"🧠 开始训练 {task} ...")
    trainer.train()
    print(f"✅ {task} 训练完成，最佳模型保存在：{task_output_dir}")

    # 验证集评估
    metrics = trainer.evaluate()
    print(f"📊 {task} 验证集评估：")
    for key, value in metrics.items():
        print(f"   - {key}: {value:.4f}")

    # 预测 test.tsv，并保存
    print(f"🔍 正在预测 {task} 的 test.tsv 数据 ...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    output_file = os.path.join(task_output_dir, "test_predictions.txt")
    os.makedirs(task_output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for label in pred_labels:
            f.write(f"{label}\n")

    print(f"📁 {task} 测试集预测结果已保存到：{output_file} ✅\n提交该文件至 GLUE 官网即可评估。")
