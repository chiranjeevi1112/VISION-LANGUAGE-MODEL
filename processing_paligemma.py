from typing import Dict,List,Optional,Union,Tuple,Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDORD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDORD_STD = [0.5,0.5,0.5]
def process_images(
        images:List[Image.Image],
        size:Dict[str,int] =None,
        resample:Image.Resampling=None,
        rescale_factor:float= None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float,List[float]]] = None
)->List[np.narray]:
    height,width = size[0],size[1]
    images = [
        resize(image=image,size=(height,width),resample = resample) for image in images
    ]

    images = [np.array(image) for image in images
              ]

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<IMAGE>"
    def __init__(self,tokenizer,num_image_tokens:int,image_size:int)
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        #follow Github for the tokenizer link gammatokenizer
        #https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer

        tokens_to_add ={"additional_special_tokens":{self.IMAGE_TOKEN}}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc(i:o4d)>" for i in range(1024)
        ]
        EXTRA_TOKENS += [
            f"<seg(i.o3d>" for i  in range(128)

        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
            self,
            text:List[str],
            images:List[Image.Image],
            padding: str ="longest",
            truncation:bool = True,
    )->dict:
        assert len(images) == 1 and len(text) ==1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size,self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor  =1/255.0
            image_mean = IMAGENET_STANDORD_MEAN
            image_std = IMAGENET_STANDORD_STD,  
        )    

        pixel_values = np.stack(pixel_values,axis=0) 
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text 
        ]