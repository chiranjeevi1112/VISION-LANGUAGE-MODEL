from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)


from safetensors import safe_open
from huggingface_hub import snapshot_download
import json, glob, os
from typing import Tuple
import torch



def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    local_dir = model_path

    if device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_dir,
            quantization_config=bnb,
            device_map="auto",    # offload overflow layers to CPU
        )
    else:
        dtype = torch.float32 if device == "cpu" else torch.float16
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_dir, torch_dtype=dtype
        ).to(device)

    processor = AutoProcessor.from_pretrained(local_dir)
    return model, processor