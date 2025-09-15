import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig

model_id = "google/paligemma-3b-pt-224"   # also try: "google/paligemma2-3b-mix-224"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto"   # offload layers to CPU if VRAM is tight
)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("test_images/vsdvsdavegvwrgb.jpg")
prompt = "<image> how many people are there in the picture and what are their names."
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

out = model.generate(**inputs, max_new_tokens=64)
print("output",processor.decode(out[0], skip_special_tokens=True))
