
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# ChatGLMForConditionalGeneration
# ~/.cache/huggingface/modules/transformers_modules/glm-4-9b-chat/modeling_chatglm.py

device = "cuda:1"
model_id = "/data0/lizhong/models/glm-4-9b-chat/ZhipuAI/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

query = "请你介绍一下东北大学？"
input_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": query}],
    add_generation_prompt=True,
    tokenize=False,
    return_tensors="pt",
    # return_dict=True
)
print(input_text)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": query}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
model.eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
