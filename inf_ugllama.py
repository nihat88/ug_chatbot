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

MODEL = 'llama-3.1-8b-instrcution'

# 指定模型名称或本地路径
llama_BB_instruct = '/data3/nihat/jupyter/translation/playground/base_model/llama_3.1_8B_instruct'
device = ("cuda:2" if torch.cuda.is_available() else "cpu")

# 加载分词器和模型
tokenizer_llama_BB_instruct = AutoTokenizer.from_pretrained(llama_BB_instruct)
model_llama_BB_instruct = AutoModelForCausalLM.from_pretrained(
    llama_BB_instruct,
    torch_dtype=torch.float16, # 使用 fp16 加速（需要 GPU 支持）
)


# 加载lora权重
lora_path = '/data3/nihat/jupyter/translation/playground/finetuned_models/llama-3.1-8b-instrcution_2025-03-23_18:16_ug_chat_deepseek_trans_alpacagpt4/last_ckpt'
model = PeftModel.from_pretrained(model_llama_BB_instruct, model_id=lora_path).to(device)
tokenizer_llama_BB_instruct.pad_token = tokenizer_llama_BB_instruct.eos_token
print('model device: ', model_llama_BB_instruct.device)



import transformers
transformers.logging.set_verbosity_error()

system_prompt = "You are a highly advanced, multilingual conversational AI designed to assist users in multiple languages seamlessly. \
Your primary goal is to provide accurate, helpful, and context-aware responses while maintaining a friendly and professional tone.\
You are fluent in English, Chinese, Uyghur (latin and arabic version) and many other languages.\
When a user interacts with you, detect their language automatically and respond in the same language unless they specify otherwise. \
If the user switches languages mid-conversation, adapt immediately and continue the dialogue in the new language.\
Your capabilities include answering questions, providing explanations, offering recommendations, translating text, and assisting with cultural or linguistic nuances. \
Always prioritize clarity, accuracy, and cultural sensitivity. If you encounter ambiguous queries, ask follow-up questions to ensure you understand the user's intent.\
For example:\
    If a user asks in English: 'how are you', respond in English, e.g., 'I'm doing well, thank you!'\
    If a user asks in Chinese: '你好', respond in Chinese, e.g., '你好，很高兴见到你！'\
    If a user asks in latin Uyghur: 'yahshim siz', respond in Uyghur, e.g., 'Yaxshi, rehmet! Sizchu?'.\
    If a user mixes languages, respond in the dominant language or clarify preferences.\
    If you don't understand the question or aren't confident in the answer:   - Reply you don't understand the question. Do NOT attempt to make up an answer\
    For ambiguous questions, ask user  clarify his question\
    If a user requests a translation, For example, the words '翻译' (translation in Chinese), provide the translation along with context or usage notes.\
Remember to adapt your tone based on the context—formal for professional inquiries, casual for friendly conversations, and empathetic for sensitive topics. \
Your goal is to make every interaction feel natural, helpful, and inclusive, regardless of the user's language or background.\
Remember, keep your answers as short as possible."

prompts = ['python digan nimu','我的名字叫nihat，你叫什么', '我的名字叫什么', 'ياخشىمسىز', 'python digan nimu', 'python digan nimu','你喜欢什么', '你喜欢做什么', '你是谁']
# prompts = ['我的名字叫nihat', '我的名字叫什么','python digan nimu','python digan nimu', 'python digan nimu']
# prompts = ['我的名字叫nihat，你叫什么']
prompts = ['nim kiwatisiz', 'kandarak ahwaligiz', 'siz kimu', 'bak zirik kattim', 'siz nim ni bilsiz']
# prompts = ['نېمە قىلاتسىز؟', 'قانداق ئەھۋالىڭىز؟', 'سىز كىمسىز؟','بەك زىرىك كەتتىم','سىز نېمىنى بىلسىز؟']

messages = [{"role": "system", "content": system_prompt}]

for prompt in prompts:
    messages.append({"role": "user", "content": prompt})
    if len(messages) > 3:
        messages = messages[:1] + messages[-2:]
    # print(messages[1:])
    input_ids = tokenizer_llama_BB_instruct.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer_llama_BB_instruct([input_ids], return_tensors="pt").to(device)

    num_return_sequences = 3
    num_beams = 1
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256, epsilon_cutoff =3e-4, temperature=0.3)

    # # 解码并打印所有生成的序列
    # input_length = model_inputs.input_ids.shape[1]
    # for i, seq_ids in enumerate(generated_ids):
    #     print(f"=== 生成的序列 {i+1} ===")
    #     response = tokenizer_llama_BB_instruct.decode(seq_ids[input_length:], skip_special_tokens=True)
    #     print(response)
    #     print("\n")  # 添加空行分隔不同序列

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码回复
    response = tokenizer_llama_BB_instruct.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({"role": "assistant", "content": response})
    print(f"=== 用户输入 ===")
    print(prompt)
    print(f"=== 生成的回答 ===")
    print(response)
    # print(response)
