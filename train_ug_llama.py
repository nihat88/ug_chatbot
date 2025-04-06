import os
device = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ["WORLD_SIZE"] = str(len(device.split(',')))
# os.environ["MASTER_PORT"] = "29509"  # 主节点端口（确保端口未被占用）

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset
from datasets import load_from_disk
from datetime import datetime
from transformers import DataCollatorWithPadding,  DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache() 

print('torch.cuda.is_available(): ', torch.cuda.is_available())
print('torch.cuda.current_device(): ', torch.cuda.current_device())
BATCH_SIZE = 8
EPOCH = 10
MAX_LENGTH = 128

MODEL = 'llama-3.1-8b-instrcution'
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
version = 'ug_chat_deepseek_trans_alpacagpt4'
output_dir = f'/data3/nihat/jupyter/translation/playground/finetuned_models/{MODEL}_{current_time}_{version}'
save_last_path = output_dir + "/last_ckpt"
print('output_dir: ', save_last_path)

# 指定模型名称或本地路径
llama_BB_instruct = '/data3/nihat/jupyter/translation/playground/base_model/llama_3.1_8B_instruct'

# 加载分词器和模型
tokenizer_llama_BB_instruct = AutoTokenizer.from_pretrained(llama_BB_instruct)
print('os.environ["CUDA_VISIBLE_DEVICES"]: ', os.environ["CUDA_VISIBLE_DEVICES"])
model_llama_BB_instruct = AutoModelForCausalLM.from_pretrained(
    llama_BB_instruct,
    torch_dtype=torch.float16, # 使用 fp16 加速（需要 GPU 支持）
)

datapath = '/data3/nihat/jupyter/translation/playground/datas/alpaca-gpt4_alapca_con'

train_data = load_from_disk(datapath)
# 打乱数据集
train_data = train_data.shuffle(seed=42)  # 设置随机种子确保可重复性
print('finish loading data')
print('length of train_data: ', len(train_data))

def process_func(example):    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer_llama_BB_instruct(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nSpeak in one of the languages: Uyghur, Chinese, or English, and respond based on the user's language choice.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer_llama_BB_instruct(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer_llama_BB_instruct.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer_llama_BB_instruct.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # target_modules=['all'],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)


tokenizer_llama_BB_instruct.pad_token = tokenizer_llama_BB_instruct.eos_token
model = get_peft_model(model_llama_BB_instruct, config)


train_data_tokenized_id = train_data.map(process_func, remove_columns=train_data.column_names)
# eval_data_tokenized_id = eval_data.map(process_func, remove_columns=eval_data.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer_llama_BB_instruct, padding=True)
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=BATCH_SIZE,
    logging_steps=10,
    num_train_epochs=EPOCH,
    save_steps=100, 
    learning_rate=1e-4,
    save_on_each_node=True,
    # gradient_checkpointing=True,
    # remove_unused_columns=False,
    save_total_limit=3,
    fp16=True,
    weight_decay=0.01,  # 添加权重衰减防止过拟合
    warmup_ratio=0.1,   # 添加 warmup，让学习率先慢后快
    lr_scheduler_type="cosine",  # 使用 cosine 学习率调度器
    seed=42,           # 设置随机种子
    deepspeed = '/data3/nihat/jupyter/translation/nllb/ds_z2_config.json',
)

# model = model.to(device)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset = train_data_tokenized_id,
    data_collator=data_collator
)   

trainer.train()
trainer.save_model(save_last_path)
print('output_dir: ', save_last_path)
print('Train Done')


