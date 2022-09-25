from torch import nn
import torch
import transformers


transformers.logging.set_verbosity_error()


class Classifier(nn.Module):
    def __init__(
        self,
        transformer,
        output_dim,
        hiden_state_dim = 768,
        dropout         = 0.5
    ):
        super(Classifier, self).__init__()

        self.transformer = transformer
        self.dropout     = nn.Dropout(dropout)
        self.linear      = nn.Linear(hiden_state_dim, output_dim)
        self.relu        = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.transformer(
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
    