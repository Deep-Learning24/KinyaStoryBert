import torch.nn as nn
import torch.nn.functional as F
import torch

import math


import logging

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        #logging.info(f'Entering Attention forward method')
        #logging.info(f'query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}, mask shape: {mask.shape if mask is not None else "None"}')

        #return self.attention(query, key, value, mask)
        try:
            #logging.info('Calculating scores')
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

            if mask is not None:
                #logging.info('Applying mask to scores')
                scores = scores.masked_fill(mask == 0, -1e9)

            #logging.info('Applying softmax to scores')
            p_attn = F.softmax(scores, dim=-1)

            if dropout is not None:
                #logging.info('Applying dropout')
                #logging.info(f'p_attn stats before dropout: min={torch.min(p_attn)}, max={torch.max(p_attn)}, mean={torch.mean(p_attn)}, std={torch.std(p_attn)}')
                p_attn = dropout(p_attn)
                #logging.info(f'p_attn stats after dropout: min={torch.min(p_attn)}, max={torch.max(p_attn)}, mean={torch.mean(p_attn)}, std={torch.std(p_attn)}')

            #logging.info('Returning from Attention forward method')
            return torch.matmul(p_attn, value), p_attn
        except Exception as e:
            logging.error(f'Error in Attention forward: {e}')
            raise

    def attention(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k.transpose(-2, -1))
    
        if attention_mask is not None:
            seq_length = w.size(-1)
    
            # Ensure attention_mask is reshaped correctly
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Now should be [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.expand(-1, -1, seq_length, seq_length)
    
            w = w.masked_fill(attention_mask == 0, float('-inf'))
    
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v),w