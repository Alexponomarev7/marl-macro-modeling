import torch
import torch.nn as nn


class AlgorithmDistillationTransformer(nn.Module):
    """
    A transformer-based model for algorithm distillation that learns to predict actions
    from state sequences across multiple tasks.

    Args:
        state_max_dim (int): Dimension of the input state space
        action_max_dim (int): Dimension of the output action space
        d_model (int, optional): Dimension of the model's internal representations. Defaults to 128
        nhead (int, optional): Number of attention heads in the transformer. Defaults to 4
        num_layers (int, optional): Number of transformer encoder layers. Defaults to 4
    """

    def __init__(
            self,
            state_max_dim: int,
            action_max_dim: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 4,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.task_embedding = nn.Embedding(1, d_model)
        self.state_embedding = nn.Linear(state_max_dim, d_model)

        # Position encoding for sequence
        # todo: maybe state number?
        self.pos_encoder = nn.Parameter(torch.zeros(1000, d_model))  # max sequence length of 1000

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            # batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_max_dim)

    def forward(self, states, task_id):
        """
        states: Tensor of shape [batch_size, seq_length, state_dim]
        task_id: Tensor of shape [batch_size], identifying which task the sequence belongs to
        """
        state_emb = self.state_embedding(states)  # [bs, seq_len, d_model]
        task_emb = self.task_embedding(task_id)  # [bs, d_model]

        inp = torch.cat([task_emb, state_emb], dim=1)  # [bs, seq_len+1, d_model]

        encoded = self.transformer(inp.transpose(0, 1))  # [seq_len+1, bs, d_model]
        encoded = encoded[1:]  # [seq_len, bs, d_model]
        encoded = encoded.transpose(0, 1)  # [bs, seq_len, d_model]
        actions_pred = self.action_head(encoded)  # [bs, seq_len, action_dim]

        return actions_pred
