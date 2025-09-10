import torch 
from torch import nn
from typing import Optional,Tuple,List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SingleVisionConfig,SiglipVisionModel


class KVCache():
    def __iniit__(self)->None:
        self.key_cache:List[torch.Tensor]=[]
        self.value_cache:List[torch.Tensor]=[]
    
    def num_items(self)->int:
        if len(self.key_cache)==0:
            return 0
        else:
            #shape pf the key_cache is [Batych_Size,Num_heads_KV,seq_len,Head_dim]
            return self.key_cache[0].shape[-2]



class GemmaConfig:
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings =8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings=max_position_embeddings
        self.hidden_size=hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads= num_key_value_heads
        self.rms_norm_eps=rms_norm_eps
        self.rope_theta=rope_theta
        self.attention_bias=attention_bias
        self.attention_dropout=attention_dropout
        self.pad_token_id=pad_token_id




class PaliGemmaConfig():
    def __init__(self,
                 vision_config = None,
                 text_config = None,
                 ignore_index =-100,
                 image_token_index = 256000,
                 vocab_size =257152,
                 projection_dim =20248,
                 hidden_size =2048,
                 pad_token_id=None,
                 **kwargs,
                 ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id= pad_token_id

        self.vision_config =SiglipVisionModel(**vision_config)
        self.text_config = text_config

        self.text_config =GemmaConfig(**text_config,pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens =(self.vision_config.image_size//self.vision_config.patch_size)**2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)



class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)   


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self,dim,max_position_embeddings=2048,base =10000,device=None):
        super().__init__()
        self.dim =dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        #calculate the theta according to the formula theta_i=base^(2i/dim) where i =0,1,2,1,3,....,dim//2
        # X
        inv_freq = 1.0/(self.base**(torch.arange(0,self.dim,2,dtype =torch.int64).float()/self.dim))
        self.register_buffer("inv_freq",tensor=inv_freq,persistent=False)

    @torch.no_grad()
    def forward(self,x,Position_ids,seq_len=None):
         # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(Position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = Position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)
    

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0        
        #number of heads =8
        # hidden_size =1024
        # head_dim =1024/8
        # wq:[1024,8*128]=[1024,1024]
        # wk:[1024,1*128]=[1024,128]
        # wv:[1024,1*128]=[1024,128]    

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states:torch.Tensor,
            attention_mask:Optional[torch.Tensor]=None,
            position_ids:Optional[torch.LongTensor]=None,
            kv_cache:Optional[KVCache]=None,
            **kwargs,
    )->Tuple[torch.tensor,Optional[torch.Tensor],Optional[Tuple[torch.Tensor]]]:
        
        bsz,q_len,_=hidden_states.size() #[batch_Size,seq_len,Hidden_siae]
        #[batch_Size,seq_len,  num_heads_q*head_dim]
        query_states=self.q_proj(hidden_states)
        #[batch_Size,seq_len,  num_heads_KV*head_dim]
        key_states = self.k_proj(hidden_states)
        #[batch_Size,seq_len,  num_heads_KV*head_dim]
        value_states = self.v_proj(hidden_states)
        #[batch_Size,Num_head_Q,seq_len,Head_Dim]
        query_states =query_states.view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
        #[batch_Size,Num_head_KV,seq_len,Head_Dim]
        key_states =key_states.view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpoe(1,2)
        #[batch_Size,Num_head_KV,seq_len,Head_Dim]
        value_states = value_states.view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        #erepeat the key  and values to match the number of heas of the query
        key_states = repeat_kv(key_states,self.num_key_value_groups)
        value_states = repeat_kv(value_states,self.num_key_value_groups)     

        #Repeat tge key and the value to match the nmber of heads of the query
        key_states = repeat_kv(key_states,self.num_key_value_groups)
        value_states = repeat_kv(value_states,self.num_key_value_groups)

        #perfom the key caculation as usual,Q*K^T/sqrt(head_dim).shape"[Batch_size,Num_H]
        attn_weights = torch.matmul(query_states,key_states.transpose(2,3))/math.sqr(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights+attn_weights

        #Apply the soft max

        #{batch_size,Num_Heads_Q,Seq_Len_Q,Seq_Len_KV}
        attn_weights = nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query_states.dtypr)
        #Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights,p=self.attention_dropout,training=self.training)
        #Multiply by the values .[Batch_Size,Num_Heads_Q,seq_len_Q,seq_len_KV]*[Btach_sie,Num_heads_KV,seq_len_KV,Head_dim]->[Batch_size]
        attn_output = torch.matmul(attn_weights,value_states)

        if attn_output.size()!= (bsz,self.num_heads,q_len,self.head_dim):
            raise ValueError(
                f"'attn_output' should be od size {(bsz,self.num_heads,q_len,self.head_dim)}, but is"
                f"{attn_output.size()}"

            )
        
        #maske sure the sequence lenght is the second dim #[btach_size,Num_heads_q,seq_len_q,head_dim]->[Batch_size,seq_len_q]
        attn_output = attn_output.transpose(1,2).contiguous()
        #concatena all the heads togethhes .[Batvh_size,seq_len_q,num_heads_q,Head_dim]->[Batch_size,seq_len_q,num_heads_Q*head_dim]
        attn_output = attn_output.view(bsz,q_len,-1)

        #multiply bt w_0 [Batch_Size mseq_len_q,hidden_state]
        attn_output = self.o_proj(attn_output)

        return attn_output,attn_weights




class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states
    


class GemmeRMSNorn(nn.Module):
    def __init__(self,dim: int, eps:float=-1e-6):
        super().__init__()
        self.eps= eps 
        self.weight=nn.Parameter(torch.zeros(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        output = self._norm(x.float())
        output =output*(1.0+self.weight.float())
        return output.type_as(x)
    

        


class GemmaModel(nn.Module):
    def __init__(self,config:GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_ids=config.pad_token_uid
        self.cvocab_size=config.vocab_Size
        self.embed_tkens=nn.Embeddinfg(config.vocab_Size,config.hifdeen_szie,self.padding_ids)
        self.layers =nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx)for layer_idx in range(config.num_hidden_layers)]

        )     
        self.norm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)

    def get_input_embedding(self):
            return self.embed_tokens
    
    def forward(
            self,
            attention_mask:Optional[torch.Tensor] = None,
            position_ids:Optional[torch.LongTensor] = None,
            inputs_embeds:Optional[torch.FloatTensor]=None,
            kv_cache:Optional[KVCache]=None,

    )->torch.FloatTensor:
        
        #[Batch_size,seq_len,hidden_sizq]
        hidden_states = inputs_embeds
        normalizer = torch.Tensor(self.config.hidden_size**0.56,dtype=hidden_states.dtype)
        hidden_states = hidden_states*normalizer

        for decoder_layer in self.layers:
            #{bstch_szie,seq_len,Hiden_size}

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,

            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
    




class GemmaForCausalLM(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config= config
        self.model =GemmaConfig(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)

    def get_input_embedding(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens_weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data        

         
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states
        

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config:PaliGemmaConfig):
        super().__init__()
        self.config= config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_mode_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 

    def tie_weight(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
            self,image_features:torch.Tensor,input_embeds:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor
    ):
        _,_,embed_dim = image_features.shape
        batch_size,sequence_lenght = input_ids.shape
        dtype,device = input_embeds.dtype,input_embeds.device

        #shape:[Vatch_size,seq_len,Hidden_Szie]
        scaled_image_features = image_features/(self.config.hidden_size**0.5)

        #combain embeddings of the image tokens tht text tokens ans mask out all paddinfg tokens
        final_embedding = torch.zeros(batch_size,sequence_lenght,embed_dim,dtype=input_embeds.dtype,device = input_embeds.device)
        
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

         # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
         # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ## create the atterntion mask

        dtype,device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items()==0:
            causal_mask= torch.full(
                (batch_szie,q_len_q_len),fill_value=0,dtype=dtype,device=device
            )
        else:
            #since we are generating tokens , the quey must be sibgel toke
            assert q_len ==1
            kv_len =kv_cache.num_items()+q_len


            causal_mask = torch.full(
                (batch_size,q_len,kv_len),fill_value=0,dtype=dtype,device=device
            )

        causal_mask = causal_mask.unsqueeze(1)    

        if kv_cache is not None and kv_cache.num_items()>0:
            positions_ids = attention_mask.cumsum(-1)[:,-1]
            if position_ids.dim() ==1:
                positions_ids=positions_ids.unsqueeze(0)
            else:
                positions_ids =(attention_mask.cumsum(-1)).masked_fill((attention_mask==0),1).to(device)
        return final_embedding,causal_mask,positions_ids        


    def forward(
        self,
        input_ids:torch.LongTensor =None,
        pixel_Values:torch.FloatTensor =None,
        attention_mask:Optional[torch.Tensor]=None,
        kv_cache:Optional[KVCache]=None,
    )->Tuple:
        assert torch.all(attention_mask==1),"The input cannot be paddes"

        #Extra the input embeddings
        #shape:(Batch_size,sseq_len,hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        #merge text and images
        #{batch_size,cahnnels,height,width} -> [Batch_size,Num_Patches,Embed_Dim]
        sewlected_imae_feature = self.vision_tower(pixel_Values.to(input_embeds.dtype))

        #{batch_size,cahnnels,height,width} -> [Batch_size,Num_Patches,Embed_Dim]
        image_features = self.multi_mode_projector(sewlected_imae_feature)

        #merge the embeddings of th text atokens and the image tokens
        input_embeds,attention_mask,position_ids = self._merge_input_ids_with_image_features(image_features,input_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs
    



    

