from typing import Tuple, List
import math
from einops import rearrange
import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Dict, Optional

torch.set_printoptions(precision=15)
class TorchQuantizer(torch.nn.Module):
    def __init__(self, 
                 bitwidth=18, 
                 int_bitwidth=8, 
                 signed=True,
                 rounding='CONVERGENT',
                 saturation='WRAP',
                 calibration=False,
                 quantize=True,
                 dtype=torch.float64):
        super(TorchQuantizer, self).__init__()
        self.bitwidth = bitwidth
        self.int_bitwidth = int_bitwidth
        self.signed = signed
        self.m = pow(2, self.bitwidth) if quantize else 1 #in calibration mode, no need to calculate m
        self.m_i = pow(2, self.int_bitwidth) if quantize else 1
        self.q = self.m / self.m_i
        self.q = float(self.q)
        self.lower_bound = -self.m/2 if self.signed else 0
        self.upper_bound = self.m/2-1 if self.signed else self.m-1
        self.rounding = rounding
        self.saturation = saturation
        self.calibration = calibration
        self.quantize = quantize
        self.max_int_bits = torch.tensor(-torch.inf)
        self.max_value = torch.tensor(-torch.inf)
        self.min_frac_bits = torch.tensor(torch.inf)
    def forward(self, x):
        if self.quantize == False:
            return x
        if self.calibration:
            x_flat = x.flatten()
            x_flat = x_flat[x_flat != 0]
            #check if x_flat is not empty
            if x_flat.nelement() > 0:
                max_int_bits = torch.max(torch.ceil(torch.log2(torch.abs(x_flat))).max())
                max_int_bits += 1 if self.signed else 0
                min_frac_bits = torch.min(torch.ceil(torch.log2(torch.abs(x_flat))).min())
                min_frac_bits += 1 if self.signed else 0
                self.max_int_bits = torch.max(max_int_bits, self.max_int_bits).int()
                self.min_frac_bits = torch.min(min_frac_bits, self.min_frac_bits).int()
            return x
        if self.rounding == 'CONVERGENT':
            if self.saturation == 'WRAP':
                qx = ((torch.round(x * self.q) - self.lower_bound) % (self.upper_bound - self.lower_bound + 1) + self.lower_bound) / self.q
            else:
                qx = torch.clamp(torch.round(x * self.q), self.lower_bound, self.upper_bound)/self.q
        else:
            if self.saturation == 'WRAP':
                qx = ((torch.trunc(x * self.q) - self.lower_bound) % (self.upper_bound - self.lower_bound + 1) + self.lower_bound) / self.q
            else:
                qx = torch.clamp(torch.trunc(x * self.q), self.lower_bound, self.upper_bound)/self.q
        # if qx == nan, raise expcetion
        if torch.isnan(qx).any():
            print("x:",x)
            raise Exception("Quantized value is NaN")
        return qx
    def forward_inplace(self, x):
        if self.quantize == False:
            return x
        if self.calibration:
            x_flat = x.flatten()
            x_flat = x_flat[x_flat != 0]
            if x_flat.nelement() > 0:
                max_int_bits = torch.max(torch.ceil(torch.log2(torch.abs(x_flat))).max())
                max_int_bits += 1 if self.signed else 0
                min_frac_bits = torch.min(torch.ceil(torch.log2(torch.abs(x_flat))).min())
                min_frac_bits += 1 if self.signed else 0
                self.max_int_bits = torch.max(max_int_bits, self.max_int_bits).int()
                self.min_frac_bits = torch.min(min_frac_bits, self.min_frac_bits).int()
                #self.max_int_bits += 1 if self.signed else 0
                #self.min_frac_bits += 1 if self.signed else 0
            return x
        if self.rounding == 'CONVERGENT':
            if self.saturation == 'WRAP':
                x.mul_(self.q).round_().sub_(self.lower_bound).remainder_(self.upper_bound - self.lower_bound + 1).add_(self.lower_bound).div_(self.q)
            else:
                x.mul_(self.q).round_().clamp_(self.lower_bound, self.upper_bound).div_(self.q)
        else:
            if self.saturation == 'WRAP':
                x.mul_(self.q).trunc_().sub_(self.lower_bound).remainder_(self.upper_bound - self.lower_bound + 1).add_(self.lower_bound).div_(self.q)
            else:
                x.mul_(self.q).trunc_().clamp_(self.lower_bound, self.upper_bound).div_(self.q)
        #x.mul_(self.q).round_()
        #x.clamp_(self.lower_bound, self.upper_bound)
        #x.round_()
        #x.div_(self.q)
        return x

class QLinear(torch.nn.Linear):
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 bias:bool=True, 
                 device=None,
                 dtype=torch.float64,
                 quant_config:dict=None,
                 calibration=False):
        super(QLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_config = quant_config
        self.calibration = calibration
        self.weight_qtzr = TorchQuantizer(**quant_config['weight'], calibration=calibration)
        self.bias_qtzr = TorchQuantizer(**quant_config['bias'], calibration=calibration)
        self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        self.output_qtzr = TorchQuantizer(**quant_config['output'], calibration=calibration)
        self.dtype = dtype
        #self.reset_parameters()
        
    #def reset_parameters(self):
    #    #reset to zero
    #    torch.nn.init.zeros_(self.weight, dtype=self.dtype)
    #    if self.bias is not None:
    #        torch.nn.init.zeros_(self.bias, dtype=self.dtype)
            
    def forward(self, x):
        qw = self.weight_qtzr(self.weight)
        qx = self.input_qtzr(x)
        qy = torch.matmul(qx, qw.t())
        if self.bias is not None:
            qy += self.bias_qtzr(self.bias)
        qy = self.output_qtzr(qy)
        return qy
    
class QFlashMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, 
                 embed_dim:int, 
                 num_heads:int, 
                 bias:bool=True, 
                 batch_first:bool=False, 
                 device=None, 
                 dtype=torch.float64,
                 quant_config:dict=None,
                 token_tile_size:int=1,
                 embed_tile_size:int=1,
                 head_tile_size:int=1,
                 max_neg_value:float=-80.0,
                 calibration=False):
        super(QFlashMultiheadAttention, self).__init__(embed_dim, 
                                                  num_heads,  
                                                  bias=bias, 
                                                  add_bias_kv=False, 
                                                  add_zero_attn=False,
                                                  kdim=None, 
                                                  vdim=None, 
                                                  batch_first=batch_first, 
                                                  device=device, 
                                                  dtype=dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scale = torch.tensor(1.0 / math.sqrt(self.head_dim))
        self.in_proj = QLinear(embed_dim, 
                               3*embed_dim, 
                               bias=bias, 
                               device=device, 
                               dtype=dtype,
                               quant_config=quant_config['in_proj'], calibration=calibration)
        self.scale_qtzr = TorchQuantizer(**quant_config['scale'], calibration=calibration)
        self.token_tile_size = token_tile_size
        self.embed_tile_size = embed_tile_size
        self.head_tile_size = head_tile_size
        self.max_neg_value = max_neg_value
        self.row_sum_qtzr = TorchQuantizer(**quant_config['row_sum'], calibration=calibration)
        self.exp_input_qtzr = TorchQuantizer(**quant_config['exp_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.exp_output_qtzr = TorchQuantizer(**quant_config['exp_output'], saturation='SAT', calibration=calibration)
        self.inv_input_qtzr = TorchQuantizer(**quant_config['inv_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.inv_output_qtzr = TorchQuantizer(**quant_config['inv_output'], saturation='SAT', calibration=calibration)
        self.attn_out_qtzr = TorchQuantizer(**quant_config['out_proj']['input'], calibration=calibration)
        self.out_proj = QLinear(embed_dim, 
                                embed_dim, 
                                bias=bias, 
                                device=device, 
                                dtype=dtype,
                                quant_config=quant_config['out_proj'], calibration=calibration)
        self.device = device
        self.dtype = dtype
        
    def forward(self, query, attn_mask=None, return_topk_idx=False, topk_ratio=1.0):
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        query = self.in_proj.input_qtzr(query)
        tgt_len, bsz, embed_dim = query.shape
        head_dim = embed_dim // self.num_heads
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        
        # with open("Q.txt", 'a') as f:
        #     np.savetxt(f, q.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        # with open("K.txt", 'a') as f:
        #     np.savetxt(f, k.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        # with open("V.txt", 'a') as f:
        #     np.savetxt(f, v.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')

        avg_cls_attention = torch.zeros((k.shape[1]-1,), dtype=torch.float64, device=q.device)
        # print(f"avg_cls_attention shape: {avg_cls_attention.shape}")

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((bsz * self.num_heads, tgt_len, 1), dtype = self.dtype, device = self.device)
        all_row_maxes = torch.full((bsz * self.num_heads, tgt_len, 1), self.max_neg_value, dtype = self.dtype, device = self.device)

        num_tiles = math.ceil(tgt_len / self.token_tile_size)
        if attn_mask is not None and attn_mask.ndim == 2:
            mask = attn_mask.bool()

        if attn_mask is None:
            col_masks = (None,) * num_tiles
            mask = (col_masks,) * num_tiles 
        else:
            mask = ((mask,) * num_tiles) if mask.shape[-2] == 1 else mask.split(self.token_tile_size, dim = -2)
            mask = tuple(((row_mask,) * num_tiles) if row_mask.shape[-1] == 1 else row_mask.split(self.token_tile_size, dim = -1) for row_mask in mask)

        B, Nt, E = q.shape
        scale = self.scale_qtzr(self.scale)
        row_splits = zip(
            q.split(self.token_tile_size, dim = -2),
            o.split(self.token_tile_size, dim = -2),
            mask,
            all_row_sums.split(self.token_tile_size, dim = -2),
            all_row_maxes.split(self.token_tile_size, dim = -2),
        )
        for i, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            col_splits = zip(
                k.split(self.token_tile_size, dim = -2),
                v.split(self.token_tile_size, dim = -2),
                row_mask
            )
            for j, (kc, vc, col_mask) in enumerate(col_splits):
                attn_weights = torch.einsum('... i d, ... j d -> ... i j', qc, kc) * scale
                if i == 0 and j > 0:  # CLS token (i=0) 對其他 tokens (j>0)
                    cls_scores = attn_weights[:, 0, 0]  # [batch*num_heads]
                    batch_size = cls_scores.shape[0] // self.num_heads
                    
                    for h in range(self.num_heads):
                        head_idx = h * batch_size
                        qk_val = cls_scores[head_idx]
                        
                        contribution = qk_val / self.num_heads
                        avg_cls_attention[j-1] += contribution

                if col_mask is not None:
                    attn_weights.masked_fill_(col_mask, -1000000)
                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(row_maxes, block_row_maxes)
                
                # 将attn_weights的值存储到文件
                # with open("attn_weights.txt", 'a') as f:
                #     np.savetxt(f, attn_weights.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                # 将new_row_maxes的值存储到文件
                # with open("new_row_maxes.txt", 'a') as f:
                #     np.savetxt(f, new_row_maxes.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                att_weights = attn_weights - new_row_maxes
                quant_att_weights = self.exp_input_qtzr(att_weights)
                exp_weights = torch.exp(quant_att_weights)
                exp_weights = self.exp_output_qtzr(exp_weights)
                
                # 将exp_weights的值存储到文件
                # with open("exp_weights.txt", 'a') as f:
                #     np.savetxt(f, exp_weights.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                if col_mask is not None:
                    exp_weights.masked_fill_(col_mask, 0.0)
                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = 1e-10)
                exp_values = torch.einsum('... i j, ... j d -> ... i d', exp_weights, vc)
                exp_row_max_diff = self.exp_input_qtzr(row_maxes - new_row_maxes)
                exp_row_max_diff = self.exp_output_qtzr(torch.exp(exp_row_max_diff))
                
                # 将exp_row_max_diff的值存储到文件
                # with open("exp_row_max_diff.txt", 'a') as f:
                #     np.savetxt(f, exp_row_max_diff.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
                
                new_row_sums = self.row_sum_qtzr(exp_row_max_diff * row_sums + block_row_sums)
                oc.mul_(exp_row_max_diff)
                oc.add_(exp_values)
                self.out_proj.input_qtzr.forward_inplace(oc)
                
                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)
                
                # 每次更新new_row_sums时将其存储到文件
                # with open("new_row_sums.txt", 'a') as f:
                #     np.savetxt(f, new_row_sums.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
            
            new_row_sums = self.inv_input_qtzr(new_row_sums)
            row_sum_inv = self.inv_output_qtzr(torch.reciprocal(new_row_sums + 1e-10))
            
            # 存储未乘上inv_row_sum的O
            # with open("O.txt", 'a') as f:
            #     np.savetxt(f, oc.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
            
            oc.mul_(row_sum_inv)
            self.out_proj.input_qtzr.forward_inplace(oc)
            
            # 将row_sum_inv的值存储到文件
            # with open("row_sum_inv.txt", 'a') as f:
            #     np.savetxt(f, row_sum_inv.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        attn_output = o.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        # with open("attn_output.txt", 'a') as f:
        #     np.savetxt(f, attn_output.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        
        if return_topk_idx:
            
            left_tokens = math.ceil(topk_ratio * avg_cls_attention.shape[0])
            
            if left_tokens > avg_cls_attention.shape[0]:
                left_tokens = avg_cls_attention.shape[0]
            
            _, topk_idx = torch.topk(avg_cls_attention, left_tokens, largest=True, sorted=True)
            print(topk_idx)
            return attn_output, topk_idx
        else:
            return attn_output
        

    
class QLayerNorm(torch.nn.LayerNorm):
    def __init__(self, 
                 normalized_shape:Tuple[int, ...],
                 quant_config:dict=None,
                 calibration=False,
                 device='cpu',
                 dtype=torch.float64,
                 ifa_head=False):
        super(QLayerNorm, self).__init__(normalized_shape, device=device, dtype=dtype)
        self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        if not ifa_head:
            self.scale_qtzr = TorchQuantizer(**quant_config['scale'], calibration=calibration)
            self.bias_qtzr = TorchQuantizer(**quant_config['bias'], calibration=calibration)
            self.output_qtzr = TorchQuantizer(**quant_config['output'], calibration=calibration)
            self.mean_qtzr = TorchQuantizer(**quant_config['mean'], calibration=calibration)
            self.var_input_qtzr = TorchQuantizer(**quant_config['var_input'], rounding='TRUNCATE', saturation='SAT', calibration=calibration)
            self.var_output_qtzr = TorchQuantizer(**quant_config['var_output'], saturation='SAT', calibration=calibration)
            self.inv_embed_dim = torch.tensor(1.0 / self.normalized_shape[-1])
            dim_int = np.ceil(np.log2(1.0/self.normalized_shape[-1]))
            self.dim_qtzr = TorchQuantizer(bitwidth=18, int_bitwidth=dim_int, signed=False, calibration=calibration)
            self.inv_embed_dim = self.dim_qtzr(self.inv_embed_dim)
    def forward(self, x):
        x = self.input_qtzr(x)
        # with open("layernorm_data.txt", 'a') as f:
        #     np.savetxt(f, x.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        xsum = x.sum(dim=-1)
        xmean = xsum * self.inv_embed_dim
        xmean = self.mean_qtzr(xmean)
        xsqr = x**2
        xsqrsum = xsqr.sum(dim=-1)
        xsqrsum = xsqrsum * self.inv_embed_dim
        xvar = xsqrsum - xmean**2
        xvar = self.var_input_qtzr(xvar)
        xvar = torch.sqrt(xvar+1e-15)
        xvar = torch.reciprocal(xvar)
        xvar = self.var_output_qtzr(xvar)
        xnorm = (x - xmean.unsqueeze(-1)) * xvar.unsqueeze(-1)
        weight = self.scale_qtzr(self.weight)
        xnorm.mul_(weight)
        bias = self.bias_qtzr(self.bias)
        xnorm.add_(bias)
        xnorm = self.output_qtzr(xnorm)
        
        # 儲存xsum、xsqrsum、xmean和xvar至相應的txt文件
        # with open("sum.txt", 'a') as f:
        #     np.savetxt(f, xsum.detach().cpu().numpy(), fmt='%.6f')
        # with open("sqrsum.txt", 'a') as f:
        #     np.savetxt(f, xsqrsum.detach().cpu().numpy(), fmt='%.6f')
        # with open("mean.txt", 'a') as f:
        #     np.savetxt(f, xmean.detach().cpu().numpy(), fmt='%.6f')
        # with open("var.txt", 'a') as f:
        #     np.savetxt(f, xvar.detach().cpu().numpy(), fmt='%.6f')
        
        # 儲存xnorm至layernorm.txt
        # with open("layernorm.txt", 'a') as f:
        #     np.savetxt(f, xnorm.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        return xnorm
    
class QFeedForward(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 hidden_dim: int,
                 bias: bool = True, 
                 activation: str = 'relu',
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float64,
                 quant_config: dict = None,
                 calibration: bool = False):
        super(QFeedForward, self).__init__()
        self.in_proj = QLinear(embed_dim, 
                               hidden_dim, 
                               bias=bias, 
                               device=device, 
                               dtype=dtype,
                               quant_config=quant_config['in_proj'], calibration=calibration)
        self.activation = activation
        self.cdf_input_qtzr = TorchQuantizer(bitwidth=12, int_bitwidth=3, rounding='TRUNCATE', saturation='SAT', calibration=calibration)
        self.cdf_output_qtzr = TorchQuantizer(bitwidth=18, int_bitwidth=0, signed=False, saturation='SAT', calibration=calibration)

        self.out_proj = QLinear(hidden_dim, 
                                embed_dim, 
                                bias=bias, 
                                device=device, 
                                dtype=dtype,
                                quant_config=quant_config['out_proj'], calibration=calibration)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        # with open("linear.txt", 'a') as f:
        #     np.savetxt(f, x.detach().cpu().numpy().reshape(-1,1), fmt='%.6f')
        if self.activation == 'relu':
            x = nn.ReLU()(x)
        elif self.activation == 'gelu':
            cdf_input = self.cdf_input_qtzr(x)
            cdf_values = 0.5 * (1 + torch.erf(cdf_input / math.sqrt(2)))
            cdf_values = self.cdf_output_qtzr(cdf_values)
            x = x * cdf_values
        x = self.out_proj(x)
        return x
    
class QTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 hidden_dim: int, 
                 dropout: float = 0.0, 
                 activation: str = 'relu', 
                 norm_first: bool = True, 
                 device: str = 'cpu', 
                 dtype: torch.dtype = torch.float64,
                 quant_config: dict = None,
                 calibration: bool = False,
                 src_mask: torch.Tensor = None,
                 enable_topk: bool = False,
                 enable_evit: bool = False,
                 enable_clc: bool = False,
                 num_clr: int = 1):
        super(QTransformerEncoderLayer, self).__init__(embed_dim, 
                                                       num_heads, 
                                                       hidden_dim, 
                                                       dropout, 
                                                       activation, 
                                                       norm_first)
        self.calibration = calibration
        self.cache_qtzr = TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=calibration, saturation='SAT')
        self.self_attn = QFlashMultiheadAttention(embed_dim,
                                                    num_heads,
                                                    device=device,
                                                    dtype=dtype,
                                                    quant_config=quant_config['self_attn'], calibration=calibration)
        self.feedforward = QFeedForward(embed_dim,
                                    hidden_dim,
                                    activation=activation,
                                    device=device,
                                    dtype=dtype,
                                    quant_config=quant_config['ffn'], calibration=calibration)
        self.norm1 = QLayerNorm(embed_dim,
                                quant_config=quant_config['norm1'], calibration=calibration)
        self.norm2 = QLayerNorm(embed_dim,
                                quant_config=quant_config['norm2'], calibration=calibration)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        #self.input_qtzr = TorchQuantizer(**quant_config['input'], calibration=calibration)
        self.src_mask = src_mask

        self.enable_topk = enable_topk
        self.enable_evit = enable_evit
        self.enable_clc = enable_clc
        self.num_clr = num_clr

        if self.enable_topk and 'topk' in quant_config:
            self.topk_input_qtzr = TorchQuantizer(**quant_config['topk']['input'], calibration=calibration)
            # print(f"TopK input config: {quant_config['topk']['input']}")
            # self.topk_output_qtzr = TorchQuantizer(**quant_config['topk']['output'], calibration=calibration)
        
        if self.enable_clc and 'clc_push' in quant_config:
            self.clc_push_input_qtzr = TorchQuantizer(**quant_config['clc_push']['input'], calibration=calibration)
            # self.clc_push_output_qtzr = TorchQuantizer(**quant_config['clc_push']['output'], calibration=calibration)
        
        if self.enable_clc and 'clc_recover' in quant_config:
            self.clc_recover_input_qtzr = TorchQuantizer(**quant_config['clc_recover']['input'], calibration=calibration)
            # self.clc_recover_output_qtzr = TorchQuantizer(**quant_config['clc_recover']['output'], calibration=calibration)
    
    def _apply_topk_pruning(self, src: torch.Tensor, topk_ratio: float = 1, topk_idx=None) -> torch.Tensor:
        
        seq_len, batch_size, embed_dim = src.shape

        if seq_len <= 1:
            return src

        left_tokens = math.ceil(topk_ratio * (seq_len - 1))
        if left_tokens != (seq_len - 1):
            assert left_tokens >= 1

            patch_tokens = src[1:, :, :]  # [seq_len-1, batch_size, embed_dim]
            patch_tokens = patch_tokens.transpose(0, 1)  # [batch_size, seq_len-1, embed_dim]
            
            if topk_idx is not None:
                topk_idx, _ = torch.sort(topk_idx)
                index = topk_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, left_tokens, embed_dim)
                x_others = torch.gather(patch_tokens, dim=1, index=index)

                if self.enable_evit:
                    all_indices = torch.arange(seq_len-1, device=src.device)
                    keep_mask = torch.zeros(seq_len-1, dtype=torch.bool, device=src.device)
                    keep_mask[topk_idx] = True
                    complement_mask = ~keep_mask
                    complement_idx = all_indices[complement_mask]  # shape: [N-1-left_tokens]
                    if complement_idx.numel() > 0:
                        compl_index = complement_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, complement_idx.numel(), embed_dim)
                        non_topk = torch.gather(patch_tokens, dim=1, index=compl_index)
                        extra_token = torch.mean(non_topk, dim=1, keepdim=True)
                    else:
                        extra_token = None

            cls_token = src[0:1, :, :]  # [1, batch_size, embed_dim]
            cls_token = cls_token.transpose(0, 1)  # [batch_size, 1, embed_dim]

            pruned_src = torch.cat([cls_token, x_others], dim=1)  # [batch_size, 1+left_tokens, embed_dim]
            if self.enable_evit and extra_token is not None:
                pruned_src = torch.cat([pruned_src, extra_token], dim=1) 

            pruned_src = pruned_src.transpose(0, 1)  # [1+left_tokens, batch_size, embed_dim]
            return pruned_src
        else:
            return src

    def _apply_clc_push(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(f"src shape: {src.shape}")
        with open('python_clc_input_data.txt', 'a') as f:
            f.write("=== Python CLC Push Input Data Log ===\n")
            seq_len, batch_size, embed_dim = src.shape
            f.write(f"seq_len={seq_len}, batch_size={batch_size}, embed_dim={embed_dim}, num_clr={self.num_clr}\n")
            
            # 記錄所有輸入數據（與 HLS 格式一致）
            for i in range(seq_len):
                for j in range(embed_dim):
                    # 假設 HLS 的 data_T::size = 1，如果不是請調整
                    f.write(f"token[{i}][{j}][0] = {src[i, 0, j]:.10f}\n")

            print(f"[CLC_PUSH] Input src dtype: {src.dtype}")
            print(f"[CLC_PUSH] First token before processing: {src[1, 0, 0]:.6f}")
            f.write(f"[CLC_PUSH] Input src dtype: {src.dtype}\n")
            f.write(f"[CLC_PUSH] First token before processing: {src[1, 0, 0]:.10f}\n")

        """
        與 topk.py 預設邏輯一致 (clc_pool_clr=False)
        """
        seq_len, batch_size, embed_dim = src.shape
        cache_tokens = []

        # 對應 HLS 的函數開頭監控
        print(f"[CLC_PUSH] Started - seq_len={seq_len}, num_clr={self.num_clr}")
        
        # GAP 計算：排除 CLS 和 CLR tokens (與 topk.py clc_pool_clr=False 一致)
        if seq_len > 1 + self.num_clr:
            # 對應 topk.py 的 curr_num_pool = x.shape[1] - self.num_clr
            # gap_tokens = src[1:seq_len-self.num_clr, :, :]  # [1:seq_len-num_clr]
            # gap = torch.mean(gap_tokens, dim=0, keepdim=True)
            # cache_tokens.append(gap)

            # 對應 HLS 的 pool_start, pool_end, pool_len
            pool_start = 1  # 不包含 CLS token
            pool_end = seq_len - self.num_clr  # 不包含 CLR tokens
            pool_len = pool_end - pool_start
            
            print(f"[CLC_PUSH] GAP calculation - pool_start={pool_start}, pool_end={pool_end}, pool_len={pool_len}")

            with open('python_clc_input_data.txt', 'a') as f:
                f.write(f"[CLC_PUSH] GAP calculation - pool_start={pool_start}, pool_end={pool_end}, pool_len={pool_len}\n")

            # gap_tokens = src[pool_start:pool_end, :, :]  # [1:seq_len-num_clr]
            gap_acc = torch.zeros(batch_size, embed_dim, dtype=src.dtype, device=src.device)
        
            # 模擬 HLS 的逐維度累加（只印第一個維度）
            print(f"[GAP_ACC] Processing {pool_len} tokens for GAP")
            # gap_acc = torch.zeros(batch_size, embed_dim, dtype=src.dtype, device=src.device)
            # for i in range(pool_len):
            #     gap_acc += gap_tokens[i, :, :]
            #     # 只監控第一個 batch 和前幾個維度
            #     if i == 0 or i == pool_len-1:
            #         print(f"[GAP_ACC] token={pool_start+i}, dim=0, acc_value={gap_acc[0, 0]:.6f}")
            with open('python_clc_input_data.txt', 'a') as f:
                for i in range(pool_len):
                    token_idx = pool_start + i
                    current_token = src[token_idx, :, :]
                    
                    # 印出每個 token 的第一個維度值來比較
                    f.write(f"[GAP_ACC] token={token_idx}, dim=0, before_acc={gap_acc[0, 0]:.10f}, adding={current_token[0, 0]:.10f}\n")
                    print(f"[GAP_ACC] token={token_idx}, dim=0, before_acc={gap_acc[0, 0]:.6f}, adding={current_token[0, 0]:.6f}")
                    
                    gap_acc += current_token
                    
                    if i == 0 or i == pool_len-1:
                        print(f"[GAP_ACC] token={token_idx}, dim=0, acc_value={gap_acc[0, 0]:.6f}")
                        f.write(f"[GAP_ACC] token={token_idx}, dim=0, acc_value={gap_acc[0, 0]:.10f}\n")

            # 對應 HLS 的正規化
            print(f"[GAP_NORM] Starting normalization, pool_len={pool_len}")
            gap = gap_acc / pool_len

            # 記錄正規化結果
            with open('python_clc_input_data.txt', 'a') as f:
                f.write(f"[GAP_NORM] Starting normalization, pool_len={pool_len}\n")
                f.write(f"[GAP_RESULT] dim=0, normalized_value={gap[0, 0]:.10f}\n")
            
            # 只印前幾個維度的結果
            print(f"[GAP_RESULT] dim=0, normalized_value={gap[0, 0]:.6f}")
            
            gap = gap.unsqueeze(0)  # 添加 sequence 維度
            cache_tokens.append(gap)

        elif seq_len > 1:
            # 如果序列太短，至少計算所有 patch tokens
            patch_tokens = src[1:, :, :]
            gap = torch.mean(patch_tokens, dim=0, keepdim=True)
            cache_tokens.append(gap)
        
        # CLR tokens：取序列的最後 num_clr 個 tokens
        if self.num_clr > 0 and seq_len > self.num_clr:
            print(f"[CLC_PUSH] Adding {self.num_clr} CLR tokens")
            clr = src[-self.num_clr:, :, :].clone()
            print(f"[CLC_PUSH] CLR token shape: {clr.shape}")
            cache_tokens.append(clr)
        
        cache_data = torch.cat(cache_tokens, dim=0) if cache_tokens else torch.empty(0, batch_size, embed_dim)
        
        # 量化前後對比
        print(f"[CLC_PUSH] Before quantization: {cache_data[0, 0, 0]:.6f}" if cache_data.numel() > 0 else "[CLC_PUSH] Empty cache")
        cache_data = self.cache_qtzr(cache_data)
        print(f"[CLC_PUSH] After quantization: {cache_data[0, 0, 0]:.6f}" if cache_data.numel() > 0 else "[CLC_PUSH] Empty cache after quant")
        
        return src, cache_data

    def _apply_clc_recover(self, src: torch.Tensor, cross_layer_cache: List[torch.Tensor]) -> torch.Tensor:
        if not cross_layer_cache:
            return src
            
        # 將所有 cache 數據 concatenate
        cached_tokens = torch.cat(cross_layer_cache, dim=0)  # [n_cached_tokens, batch_size, embed_dim]
        # cached_tokens = self.cache_qtzr(cached_tokens)

        # print('src shape:', src.shape)
        # print('cached_tokens shape:', cached_tokens.shape)
        # 將 cache 數據添加到當前序列的末尾
        recovered_src = torch.cat([src, cached_tokens], dim=0)
        
        return recovered_src

    def forward(self, 
                src: torch.Tensor, 
                cross_layer_cache: Optional[List[torch.Tensor]] = None,
                layer_idx: int = 0,
                is_reduction_layer: bool = False,
                is_recovery_layer: bool = False,
                is_push_layer: bool = False,
                keep_rate: float = 1.0,
                src_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        cache_data = None

        if self.norm_first:
            src = self.norm1.input_qtzr(src) #add input quantizer
            src_norm = self.norm1(src)
            if is_reduction_layer and self.enable_topk:
                src2, topk_idx = self.self_attn(src_norm, attn_mask=src_mask, return_topk_idx=True, topk_ratio=keep_rate)
            else:
                src2 = self.self_attn(src_norm, attn_mask=src_mask)
            src = src + self.dropout(src2)

            if is_reduction_layer and self.enable_topk:
                src = self.topk_input_qtzr(src)
                src = self._apply_topk_pruning(src, keep_rate, topk_idx=topk_idx)
            
            src = self.norm2.input_qtzr(src) #add input quantizer
            src_norm = self.norm2(src)
            src2 = self.feedforward(src_norm)
            src = src + self.dropout(src2)

            if is_recovery_layer and self.enable_clc and cross_layer_cache:
                src = self.clc_recover_input_qtzr(src)
                src = self._apply_clc_recover(src, cross_layer_cache)

            if self.enable_clc and is_push_layer:
                src = self.clc_push_input_qtzr(src)
                src, cache_data = self._apply_clc_push(src)

        else:
            if is_reduction_layer and self.enable_topk:
                src2, topk_idx = self.self_attn(src, attn_mask=src_mask, return_topk_idx=True, topk_ratio=keep_rate)
            else:
                src2 = self.self_attn(src, attn_mask=src_mask)
            src = src + self.dropout(src2)
            src = self.norm1(src)

            if is_reduction_layer and self.enable_topk:
                src = self.topk_input_qtzr(src)
                src = self._apply_topk_pruning(src, keep_rate, topk_idx=topk_idx)

            src2 = self.feedforward(src)
            src = src + self.dropout(src2)
            src = self.norm2(src)

            # CLC operations
            if is_recovery_layer and self.enable_clc and cross_layer_cache:
                src = self.clc_recover_input_qtzr(src)
                src = self._apply_clc_recover(src, cross_layer_cache)

            if self.enable_clc and is_push_layer:
                src = self.clc_push_input_qtzr(src)
                src, cache_data = self._apply_clc_push(src)

        return src, cache_data

class QTransformerEncoder(nn.TransformerEncoder):
    def __init__(self,
                 encoder_layer: List[QTransformerEncoderLayer],
                 num_layers: int,
                 norm: QLayerNorm,
                 input_qtzr: TorchQuantizer,
                 args,
                 dtype: torch.dtype = torch.float64):
        super(QTransformerEncoder, self).__init__(encoder_layer[0], num_layers, norm)
        self.layer_list = encoder_layer
        self.norm = norm
        self.input_qtzr = input_qtzr
        self.dtype = dtype

        # 從 args 中獲取配置
        # self.enable_topk = getattr(args, 'enable_topk', False)
        self.enable_clc = getattr(args, 'clc', False)
        self.num_clr = getattr(args, 'num_clr', 1)
        self.ifa_head = getattr(args, 'ifa_head', False)
        
        # TopK 和 CLC 相關配置
        self.reduction_loc = getattr(args, 'reduction_loc', [])
        keep_rates = getattr(args, 'keep_rate', [])
        
        if keep_rates:
            if len(keep_rates) == 1 and len(self.reduction_loc) > 1:
                # 如果只提供一個 keep_rate，則所有 reduction layer 都使用這個值
                self.keep_rates = keep_rates * len(self.reduction_loc)
            elif len(keep_rates) == len(self.reduction_loc):
                # 如果長度匹配，直接使用
                self.keep_rates = keep_rates
            else:
                # 如果長度不匹配，補齊或截斷
                self.keep_rates = (keep_rates * ((len(self.reduction_loc) // len(keep_rates)) + 1))[:len(self.reduction_loc)]
        else:
            # 如果沒有提供 keep_rates，使用預設值 1.0
            self.keep_rates = [1.0] * len(self.reduction_loc)

        # 計算 recovery layers
        clc_recover_at_last = getattr(args, 'clc_recover_at_last', True)
        if self.enable_clc:
            if clc_recover_at_last:
                self.recovery_layers = self.reduction_loc + [num_layers - 2] if num_layers >= 2 else self.reduction_loc
            else:
                self.recovery_layers = self.reduction_loc
        else:
            self.recovery_layers = []

        # print(f"Enable TopK: {self.enable_topk}")
        print(f"Enable CLC: {self.enable_clc}")
        print(f"Number of CLR tokens: {self.num_clr}")
        print(f"Reduction locations: {self.reduction_loc}")
        print(f"Recovery layers: {self.recovery_layers}")
        print(f"Keep rates: {self.keep_rates}")

    def forward(self, 
                src: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        
        with open('python_clc_input_data.txt', 'w') as f:
            f.write("")
        src = self.input_qtzr(src)
        output = src
        print('input: ', output[0][0][0:3])
        cross_layer_cache = []

        for i, mod in enumerate(self.layer_list):
            is_reduction_layer = i in self.reduction_loc
            is_recovery_layer = i in self.recovery_layers
            if self.recovery_layers:
                is_push_layer = i < self.recovery_layers[-1]
            else:
                is_push_layer = False

            keep_rate = 1.0  # 預設值改為 1.0
            if is_reduction_layer:
                reduction_idx = self.reduction_loc.index(i)
                keep_rate = self.keep_rates[reduction_idx]
            
            # CLC recover 時清空 cache (避免重複使用)
            current_cache = cross_layer_cache.copy() if is_recovery_layer else None
            
            output, cache_data = mod(output, 
                                   cross_layer_cache=cross_layer_cache,
                                   layer_idx=i,
                                   is_reduction_layer=is_reduction_layer,
                                   is_recovery_layer=is_recovery_layer,
                                   is_push_layer=is_push_layer,
                                   keep_rate=keep_rate,
                                   src_mask=mask)
            # 如果當前層執行了 CLC recover，在 recovery 後清空 cache
            if is_recovery_layer and current_cache is not None:
                cross_layer_cache = []

            if cache_data is not None and self.enable_clc and is_push_layer:
                cross_layer_cache.append(cache_data)
            #     print(f"[DEBUG] Layer {i}: Added cache, total cache_size={len(cross_layer_cache)}")
            # else:
            #     print(f"[DEBUG] Layer {i}: Not collecting cache (i >= recovery_layers[-1])")

        if not self.ifa_head:
            output = self.norm(output)
        else:
            output = self.norm.input_qtzr(output)
            
        return output
    
    def transfer_weights(self, 
                         model: nn.Module):
        for i, layer in enumerate(self.layer_list):
            layer.norm1.weight.data = model.transformer_encoder.layers[i].norm1.weight.type(self.dtype)
            layer.norm1.bias.data = model.transformer_encoder.layers[i].norm1.bias.type(self.dtype)
            layer.norm2.weight.data = model.transformer_encoder.layers[i].norm2.weight.type(self.dtype)
            layer.norm2.bias.data = model.transformer_encoder.layers[i].norm2.bias.type(self.dtype)
            layer.self_attn.in_proj.weight.data = model.transformer_encoder.layers[i].self_attn.in_proj_weight.type(self.dtype)
            layer.self_attn.in_proj.bias.data = model.transformer_encoder.layers[i].self_attn.in_proj_bias.type(self.dtype)
            layer.self_attn.out_proj.weight.data = model.transformer_encoder.layers[i].self_attn.out_proj.weight.type(self.dtype)
            layer.self_attn.out_proj.bias.data = model.transformer_encoder.layers[i].self_attn.out_proj.bias.type(self.dtype)
            layer.feedforward.in_proj.weight.data = model.transformer_encoder.layers[i].linear1.weight.type(self.dtype)
            layer.feedforward.in_proj.bias.data = model.transformer_encoder.layers[i].linear1.bias.type(self.dtype)
            layer.feedforward.out_proj.weight.data = model.transformer_encoder.layers[i].linear2.weight.type(self.dtype)
            layer.feedforward.out_proj.bias.data = model.transformer_encoder.layers[i].linear2.bias.type(self.dtype)
        if not model.ifa_head:
            if hasattr(self.norm, 'weight') and hasattr(self.norm, 'bias'):
                self.norm.weight.data = model.transformer_encoder.norm.weight.type(self.dtype)
                self.norm.bias.data = model.transformer_encoder.norm.bias.type(self.dtype)
            else:
                print("Warning: QTransformerEncoder.norm does not have weight/bias, but source norm is not Identity")
        else:
            print("Source transformer norm is Identity, skipping norm weight transfer")


def calibrate_transformer(qmodel: QTransformerEncoder, 
                          quant_config: dict, 
                          calibration_data: torch.Tensor,
                          calibration_mask: torch.Tensor = None
                          ) -> dict:
    with torch.no_grad():
        qmodel.eval()
        qy = qmodel(calibration_data, mask=calibration_mask)

        for i, layer in enumerate(qmodel.layer_list):
            is_reduction_layer = i in qmodel.reduction_loc
            is_recovery_layer = i in qmodel.recovery_layers
            if qmodel.recovery_layers:
                is_push_layer = i < qmodel.recovery_layers[-1]
            else:
                is_push_layer = False
            #print("Calibrating layer:", id(layer.norm1.input_qtzr.max_int_bits))
            #print("Calibrating:", layer.norm1.input_qtzr.max_int_bits)
            quant_config[i]['norm1']['input']['int_bitwidth'] = layer.norm1.input_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['mean']['int_bitwidth'] = layer.norm1.mean_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['scale']['int_bitwidth'] = layer.norm1.scale_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['bias']['int_bitwidth'] = layer.norm1.bias_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['output']['int_bitwidth'] = layer.norm1.output_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['var_input']['int_bitwidth'] = layer.norm1.var_input_qtzr.max_int_bits.item()
            quant_config[i]['norm1']['var_output']['int_bitwidth'] = layer.norm1.var_output_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['input']['int_bitwidth'] = layer.norm2.input_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['mean']['int_bitwidth'] = layer.norm2.mean_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['scale']['int_bitwidth'] = layer.norm2.scale_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['bias']['int_bitwidth'] = layer.norm2.bias_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['output']['int_bitwidth'] = layer.norm2.output_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['var_input']['int_bitwidth'] = layer.norm2.var_input_qtzr.max_int_bits.item()
            quant_config[i]['norm2']['var_output']['int_bitwidth'] = layer.norm2.var_output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['input']['int_bitwidth'] = layer.self_attn.in_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['weight']['int_bitwidth'] = layer.self_attn.in_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['bias']['int_bitwidth'] = layer.self_attn.in_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['in_proj']['output']['int_bitwidth'] = layer.self_attn.in_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['input']['int_bitwidth'] = layer.self_attn.out_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['weight']['int_bitwidth'] = layer.self_attn.out_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['bias']['int_bitwidth'] = layer.self_attn.out_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['out_proj']['output']['int_bitwidth'] = layer.self_attn.out_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['row_sum']['int_bitwidth'] = layer.self_attn.row_sum_qtzr.max_int_bits.item()
            #quant_config[i]['self_attn']['exp_input']['int_bitwidth'] = layer.self_attn.exp_input_qtzr.max_int_bits.item()
            #quant_config[i]['self_attn']['exp_output']['int_bitwidth'] = layer.self_attn.exp_output_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['inv_input']['int_bitwidth'] = layer.self_attn.inv_input_qtzr.max_int_bits.item()
            quant_config[i]['self_attn']['inv_output']['int_bitwidth'] = layer.self_attn.inv_output_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['input']['int_bitwidth'] = layer.feedforward.in_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['weight']['int_bitwidth'] = layer.feedforward.in_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['bias']['int_bitwidth'] = layer.feedforward.in_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['in_proj']['output']['int_bitwidth'] = layer.feedforward.in_proj.output_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['input']['int_bitwidth'] = layer.feedforward.out_proj.input_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['weight']['int_bitwidth'] = layer.feedforward.out_proj.weight_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['bias']['int_bitwidth'] = layer.feedforward.out_proj.bias_qtzr.max_int_bits.item()
            quant_config[i]['ffn']['out_proj']['output']['int_bitwidth'] = layer.feedforward.out_proj.output_qtzr.max_int_bits.item()

            if is_reduction_layer and layer.enable_topk:
                quant_config[i]['topk']['input']['int_bitwidth'] = layer.topk_input_qtzr.max_int_bits.item()
                # quant_config[i]['topk']['output']['int_bitwidth'] = layer.topk_output_qtzr.max_int_bits.item()
            
            if is_recovery_layer and layer.enable_clc:
                quant_config[i]['clc_recover']['input']['int_bitwidth'] = layer.clc_recover_input_qtzr.max_int_bits.item()
                # quant_config[i]['clc_recover']['output']['int_bitwidth'] = layer.clc_recover_output_qtzr.max_int_bits.item()

            if is_push_layer and layer.enable_clc:
                quant_config[i]['clc_push']['input']['int_bitwidth'] = layer.clc_push_input_qtzr.max_int_bits.item()
                # quant_config[i]['clc_push']['output']['int_bitwidth'] = layer.clc_push_output_qtzr.max_int_bits.item()

        
        quant_config['norm']['input']['int_bitwidth'] = qmodel.norm.input_qtzr.max_int_bits.item()
        if not qmodel.ifa_head:
            quant_config['norm']['mean']['int_bitwidth'] = qmodel.norm.mean_qtzr.max_int_bits.item()
            quant_config['norm']['scale']['int_bitwidth'] = qmodel.norm.scale_qtzr.max_int_bits.item()
            quant_config['norm']['bias']['int_bitwidth'] = qmodel.norm.bias_qtzr.max_int_bits.item()
            quant_config['norm']['output']['int_bitwidth'] = qmodel.norm.output_qtzr.max_int_bits.item()
            quant_config['norm']['var_input']['int_bitwidth'] = qmodel.norm.var_input_qtzr.max_int_bits.item()
            quant_config['norm']['var_output']['int_bitwidth'] = qmodel.norm.var_output_qtzr.max_int_bits.item()

    return quant_config