import os
import csv  # 新增CSV模块

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import PeftModel
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

# 基础路径配置
base_model_path = "/root/nfs/LLMDatasetsAndModels/models/bert-base-cased"
base_output_path = "/root/nfs/WCYLora/fedlora/matrix-layer-fedlora/output/yelp_bert_rora_top4_50_20250425104854"

# 初始化结果文件 (新增部分)
result_path = "result.csv"
with open(result_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "accuracy", "F1_score"])

# 加载测试数据
dataset = load_dataset(
    "/root/nfs/LLMDatasetsAndModels/datasets/yelp_review_full", split="test"
)
test_dataset = (
    dataset.shard(num_shards=2, index=1).shuffle(seed=2024).select(range(500))
)
class_names = dataset.features["label"].names
id2label = {i: label for i, label in enumerate(class_names)}

# 初始化基础组件
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


tokenized_eval_dataset = test_dataset.map(preprocess_function, batched=True)

# 遍历检查点
for checkpoint in range(1, 51):  # 1到25
    checkpoint_path = f"{base_output_path}/checkpoint-{checkpoint}"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint} not found, skipping...")
        continue

    print(f"\n{'='*40}")
    print(f"Evaluating checkpoint-{checkpoint}")
    print(f"{'='*40}")

    try:
        # 加载模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path, id2label=id2label
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()  # 合并LoRA权重

        # 配置Trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./",
                per_device_eval_batch_size=16,
                report_to="none",  # 禁用日志上报
            ),
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
        )

        # 评估模型
        predictions = trainer.predict(tokenized_eval_dataset)
        pred_labels = predictions.predictions.argmax(-1)
        true_labels = predictions.label_ids

        # 计算指标
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted")

        # 写入CSV结果 (新增部分)
        with open(result_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([checkpoint, round(accuracy, 4), round(f1, 4)])

        print(f"Checkpoint-{checkpoint} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # 释放显存
        del model, base_model, trainer
        import torch

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing checkpoint-{checkpoint}: {str(e)}")
