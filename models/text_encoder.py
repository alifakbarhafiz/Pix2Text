import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

tokenizer = None
text_model = None


def init_text_models(device):
    """Load DistilBERT; call this before using TextEncoder."""
    global tokenizer, text_model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    return tokenizer, text_model


class TextEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.linear = nn.Linear(768, out_dim)

    def forward(self, texts):
        dev = next(text_model.parameters()).device
        tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(dev)
        out = text_model(**tok).last_hidden_state[:, 0]
        return self.linear(out)
