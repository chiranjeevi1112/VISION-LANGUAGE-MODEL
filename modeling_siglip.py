import torch
import numpy as np
import cv2
#Text encodeder and Vision Encoder both are kind of transformer model
#contrastive learning we have the lsit of texts and the lists of images both embedding are there anf their dot product 
#Has to be equal to 1 in the diagonal once remaining has to be close to zero
#CLIP bove flatten featuresa and send the whole
#
from typing import Optional,Tuple
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)


class SingleVisionConfig:
    def __init__(self,
                 hidden_size = 768,
                 intermediate_size = 3072,
                 num_hidden_layers = 12,
                 num_attention_heads = 12,
                 num_channels =3,
                 image_size = 224,
                 patch_size = 16,
                 layer_more_eps = 1e-6,
                 attention_dropout = 0.0,
                 num_image_tokens:int=None,
                 **kwargs):

                super().__init__()
                self.hidden_size = hidden_size
                self.intermediate_size = intermediate_size
                self.num_hidden_layers = num_hidden_layers
                self.num_attention_heads = num_attention_heads
                self.num_channels = num_channels
                self.image_size = image_size
                self.patch_size = patch_size
                self.layer_norm_eps = layer_more_eps
                self.attention_dropout = attention_dropout
                self.num_imahe_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
       def __init__(self,config:SingleVisionConfig):
              super().__init_()
              self.config = config
              self.embed_dim = config.hidden_size
              self.image_size = config.image_size
              self.patch_size = config.patch_size

              self.patch_embedding = nn.Conv2d(
                     in_channels = config.num_channels,
                     out_channels = config.embed_dim,
                     kernel_size = self.patch_size,
                     stride = self.patch_size,
                     padding = "valid", #No padding is added
              )

              self.num_patches = (self.image_size//self.patch_size)**2
              self.num_positions = self.num_patches
              self.positions_embedding = nn.Embedding(self.num_positions,
                                                      self.embed_dim)
              self.register_buffer(
                     "position_ids",
                     torch.arange(self.num_positions).expand(1,-1),
                     persistent=False,
              )
       def forward(self,pixel_values:torch.FloatTensor)->torch.tensor:
              _, _, height, width = pixel_values.shape #[Batch size,channels,height,width]
              #convolve the patch_size kernel over the image with no overlapping patches since 
              #the output of the convolution will have the shape [btchsize,Embed_Dim,num_patches_H
              patch_embed = self.patch_embedding(pixel_values)
              embeddings = patch_embed.flatten(2)
              embeddings = embeddings.transpose(1,2)
              embeddings = embeddings+self.positions_embedding(self.position_ids)
              return embeddings

#layer nomralization is because big change in the input layer leads to big chage in the output layer 
#and big change in the loss layer and leads to change in the gadient layer leads to chage in the weights
#this is the reason we need to do the layer normalization
#first solution is batch normlaization which basically normlaize the batchs
#problem is we need more batch size to work perfectly and the normalized vlues depends on the input of the bathes
#layer normalization is something we take the input image or vector or something
#calcualte the mean and the varience of the entire image or vector and normalize it
class SiglipMLP(nn.Module):
       def __init__(self,config):
              super().__init__()
              self.config = config
              self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size)
              self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)

       def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
              hidden_states = self.fc1(hidden_states)
              hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
              hidden_states = self.fc2(hidden_states)

              return hidden_states
            
               
class SiglipAttention(nn.Module):
       def __init__(self,config):
              super().__init__()
              self.config = config
              self.embed_dim = config.hidden_size
              self.num_heads = config.num_attention_heads
              self.head_dim = self.embed_dim//self.num_heads
              self.scale = self.head_dim**-0.5
              self.dropout = config.attention_dropout

              self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
              self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
              self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
              self.out_proj = nn.Linear(self.embed_dim,self.embed_dim)
       def forward(self,
                   hidden_states:torch.tensor,)->Tuple[torch.tensor,Optional[torch.Tensor]]:
              batch_size, seq_len,_ = hidden_states.size()
              query_states = self.q_proj(hidden_states)
              key_states = self.k_proj(hidden_states)
              value_States = self.v_proj(hidden_states)
              
              query_states = query_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
              key_states = key_states.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

              value_States = value_States.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
              
              attn_weights = (torch.matmul(query_states,key_states.transpose(2,3))*self.scale)
              
              if attn_weights.size()!= (batch_size,self.num_heads,seq_len,seq_len):
                     raise ValueError(
                            f"Attention weight should be of size {(batch_size,self.num_heads,seq_len,seq_len)},but is "
                            f"{attn_weights.size()}"
                     )
              
              attn_weights = nn.functional.softmax(attn_weights,dim =1,dtype=torch.float32).to(query_states.dtype)

              attn_output = torch.matmul(attn_weights,value_States)

              if attn_output.size()!=(batch_size,self.num_heads,seq_len,self.head_dim):
                     raise ValueError(
                            f"attn_output should be of size {(batch_size,self.num_heads,seq_len,self.head_dim)},but is"
                            f"{attn_output.size()}" 
                     )
              attn_output = attn_output.transpose(1,2).contiguous()

              attn_output = attn_output.reshape(batch_size,seq_len,self.embed_dim)

              attn_output = self.out_proj(attn_output)

              return attn_output,attn_weights
       



class SiglipEncoderLayer(nn.Module):
       def __init__(self,config:SingleVisionConfig):
              super().init__()
              self.embed_dim = config.hidden_size
              self.self_attn = SiglipAttention(config)
              self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps = config.layer_more_eps)
              self.mlp = SiglipMLP(config)
              self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)

       def forward(
                  self,
                   hidden_states:torch.Tensor
                   )->torch.Tensor:
              
              residual = hidden_states
              hidden_states = self.layer_norm1(hidden_states)
              hidden_states = self.self_attn(hidden_states = hidden_states)
              hidden_states = residual + hidden_states

              residual = hidden_states
              
              hidden_states = self.layer_norm2(hidden_states)
              hidden_states = self.mlp(hidden_states) 

              hidden_states = residual + hidden_states

              return hidden_states
         

class SiglipEncoder(nn.Modules):
       def __init__(self,config:SingleVisionConfig):
              super().__init__()
              self.config = config
              self.layers = nn.ModuleList(
                     [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
              )
       def forward(
           self,
           input_embeds:torch.Tensor
       )->torch.Tensor:
              hidden_states = input_embeds

              for encoder_layer in self.layers:
                     hidden_states = encoder_layer(hidden_states)

              return hidden_states              

       

class siglipVisionTransformer(nn.Module):
       def __init__(self,config:SingleVisionConfig):
              super().__init__()
              self.config = config
              embad_dim = config.hidden_size

              self.embeddings = SiglipVisionEmbeddings(config)
              self.encoder = SiglipEncoder(config)
              self.post_layernorm = nn.LayerNorm(embad_dim,eps = config.layer_norm_eps)
        
       def forward(self,pixel_values:torch.Tensor)->torch.tensor:
              hidden_states = self.embeddings(pixel_values)
              last_hidden_state = self.encoder(inputs_embed=hidden_states)
              last_hidden_state = self.post_layernorm(layer_hidden_State)

              return last_hidden_state


class SiglipVisionModel(nn.Module):
        
        def __init__(self,config:SingleVisionConfig):
            super().__init__()
            self.config = config
            self.vision_model = SiglipVisionModel(config)
         
        def forward(self,pixel_values)->Tuple:
               
               return self.vision_model(pixel_values=pixel_values)
        



        
                
                  
        