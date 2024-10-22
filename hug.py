from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch

model_id = "D:\python_code\LVMS\paligemma-3b-pt-224"
device = "cuda:0"
dtype = torch.bfloat16

url = 'D:\python_code\LVMS\PaliGemma\img.png'
image = Image.open(url)


quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "i overview the image"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print( "check",decoded)


