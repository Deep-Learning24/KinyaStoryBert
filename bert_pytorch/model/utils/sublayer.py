import torch.nn as nn
from .layer_norm import LayerNorm


import logging

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #logging.info(f'Entering SublayerConnection forward method')
        #logging.info(f'x shape: {x.shape}')

        try:
            #logging.info('Applying layer norm and sublayer')
            sublayer_output = sublayer(self.norm(x))

            #logging.info('Applying dropout and adding to original input')
            output = x + self.dropout(sublayer_output)

            #logging.info('Returning from SublayerConnection forward method')
            return output
        except Exception as e:
            logging.error(f'Error in SublayerConnection forward: {e}')
            raise
