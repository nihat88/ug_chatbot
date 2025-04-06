import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"

from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import transformers
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset
from datasets import load_from_disk
from datetime import datetime
from transformers import DataCollatorWithPadding,  DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache() 
import json
transformers.logging.set_verbosity_error()

MODEL = 'llama-3.1-8b-instrcution'
# 指定模型名称或本地路径
llama_BB_instruct = '/root/autodl-tmp/basemodel/meta-llama/Llama-3.1-8B-Instruct'

cuda_device = "cuda:0"
device = (cuda_device if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
tokenizer_llama_BB_instruct = AutoTokenizer.from_pretrained(llama_BB_instruct)
model_llama_BB_instruct = AutoModelForCausalLM.from_pretrained(
    llama_BB_instruct,
    # load_in_8bit=True,
    torch_dtype=torch.float16, # 使用 fp16 加速（需要 GPU 支持）
    # use_cache=False # 禁用缓存（避免显存不足）
)


# 推理
lora_path = '/root/autodl-tmp/project_250406/0326_adapter'

# 加载lora权重
# model = model_llama_BB_instruct.to(device)
model = PeftModel.from_pretrained(model_llama_BB_instruct, model_id=lora_path).to('cuda')

print('model device: ', model_llama_BB_instruct.device)
tokenizer_llama_BB_instruct.pad_token = tokenizer_llama_BB_instruct.eos_token

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    messages = data['messages']

    def chat_with_model(messages):
        print('messages  :',messages)
        # 生成模型输入
        input_ids = tokenizer_llama_BB_instruct.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer_llama_BB_instruct([input_ids], return_tensors="pt").to(device)
        # 生成回复
        # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256, epsilon_cutoff =3e-4, temperature=0.3)
        # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256, epsilon_cutoff =3e-4)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # 解码回复
        response = tokenizer_llama_BB_instruct.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    response = chat_with_model(messages)
    print(f"Assistant: {response}")
    json_str = json.dumps(response, ensure_ascii=False)
    return json_str


if __name__ == '__main__':
    port = 24000
    app.run(debug=False, host='0.0.0.0', port=int(port))

