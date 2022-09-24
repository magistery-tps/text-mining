from torch import nn
from transformers import BertModel
import torch
import transformers


transformers.logging.set_verbosity_error()


class BertClassifier(nn.Module):
    def __init__(
        self,
        output_dim,
        model           ='bert-base-cased',
        hiden_state_dim = 768,
        dropout         = 0.5,
        cache_dir       = "../../cache"
    ):
        super(BertClassifier, self).__init__()

        self.bert    = BertModel.from_pretrained(model, use_cache=True, cache_dir=cache_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hiden_state_dim, output_dim)
        self.relu    = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids      = input_id,
            attention_mask = mask,
            return_dict    = False
        )

        dropout_output   = self.dropout(pooled_output)
        linear_output    = self.linear(dropout_output)
        final_layer      = self.relu(linear_output)

        return final_layer
    

    def load(self, path): self.load_state_dict(torch.load(path))

    def save(self, path): torch.save(self.state_dict(), path)
    