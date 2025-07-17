import torch
from transformers import BertTokenizer
from model import BERT_CRF
import json

# Load labels
with open("labels.json", "r") as f:
    LABELS = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BERT_CRF.from_pretrained("outputs/")
model.to(device)
model.eval()

def predict_ner(text):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer.encode_plus(tokens, return_tensors="pt", is_split_into_words=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)
        predicted_ids = output[0]

    predicted_labels = [LABELS[i] for i in predicted_ids[0]]
    token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return list(zip(token_list, predicted_labels))
