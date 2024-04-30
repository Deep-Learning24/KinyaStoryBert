import math
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput,BertIntermediate,BertOutput


class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Compute attention scores
        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (causal mask for autoregressive behavior)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Apply softmax to get attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, mixed_value_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# Modify the BertSelfAttention module in the model
class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None, output_attentions=False):
        self_outputs = self.self(input_tensor, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# Replace the BertSelfAttention module in the model's encoder layers
class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs

# Create the custom model
class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert.encoder.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

# Load the pre-trained model
config = BertConfig.from_pretrained("jean-paul/KinyaBERT-large")
model = CustomBertForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large", config=config)

# Print the modified model architecture
print(model)
