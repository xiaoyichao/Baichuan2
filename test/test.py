import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


tokenizer = AutoTokenizer.from_pretrained("/ssd1/share/test/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/ssd1/share/test/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/ssd1/share/test/Baichuan2-7B-Chat")
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)

prompt_outline_yuanwen2mk = '''给出大纲与原文，帮
我根据原文生成大纲下对应的内容，这些内容最后会展示在ppt中，如果在'''


# from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/ssd1/share//Baichuan2-13B-Base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/ssd1/share//Baichuan2-13B-Base", device_map="auto", trust_remote_code=True)
# inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
# inputs = inputs.to('cuda:0')
# pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
# print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))