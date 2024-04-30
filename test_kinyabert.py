from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("jean-paul/KinyaBERT-large")

model = AutoModelForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large")

input_text = "Ejo ndikwiga nagize abashyitsi baje kunsura. Ndashaka kubona niba bazakwiga cyangwa se bazasura. [MASK] ni umwihariko w'abashyitsi."
encoded_input = tokenizer(input_text, return_tensors='pt',truncation=True, padding='max_length', max_length=128)

# Create the labels by shifting the input_ids to the right
labels = encoded_input["input_ids"].clone()

# Shift the inner list to the right
labels[0, :-1] = encoded_input["input_ids"][0, 1:]


# Set the first token of the inner list to -100
labels[0, -1] = tokenizer.pad_token_id


# Set the last token of the inner list to the pad_token_id
labels[:, -1] = -100


# Squeeze the tensors and return them
inputs = {key: tensor.squeeze(0) for key, tensor in encoded_input.items()}
inputs["labels"] = labels.squeeze(0)

# Print the input dictionary
print("Input dictionary:")
print(inputs)

print(encoded_input)
output = model(**encoded_input)
print("Keys:", output.keys())
print("Output shape:", output.logits.shape)
print(output)



# Get the logits from the model's output
logits = output.logits
loss = output.loss
print("Loss:", loss)



# Convert the labels to tensor
labels = torch.tensor(labels)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# Compute the loss
loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

print("Loss:", loss.item())
# # Get the token IDs of the most probable tokens
predicted_token_ids = torch.argmax(logits, dim=-1)

# # Convert the token IDs back into strings
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0])
print("Predicted tokens:", predicted_tokens)

# Join the tokens into a single string
predicted_text = ' '.join(predicted_tokens)
print("Predicted text:", predicted_text)

# # Replace the masked tokens in the input text with the predicted tokens
# for i, token in enumerate(predicted_tokens):
#     if token == "[MASK]":
#         input_text = input_text.replace("[MASK]", predicted_tokens[i], 1)

# print("Generated text:", input_text)

