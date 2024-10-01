from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

# GLM-4-9B-Chat-1M
# max_model_len, tp_size = 1048576, 4

# GLM-4-9B-Chat
# 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
max_model_len, tp_size = 1024, 1
model_id = "/data0/lizhong/models/glm-4-9b-chat/ZhipuAI/glm-4-9b-chat"
prompt = [{"role": "user", "content": "请你介绍一下东北大学"}]

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm = LLM(
    model=model_id,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
    # enable_chunked_prefill=True,
    # max_num_batched_tokens=8192
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(
    temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

inputs = tokenizer.apply_chat_template(
    prompt, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)


# https://huggingface.co/THUDM/glm-4-9b#:~:text=GLM-4V-9B
