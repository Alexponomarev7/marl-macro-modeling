import torch
import torch.nn as nn

class AlgorithmDistillationTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, num_tasks, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.task_embedding = nn.Embedding(num_tasks, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, states, task_id):
        """
        states: Tensor of shape [batch_size, seq_length, state_dim]
        task_id: Tensor of shape [batch_size], identifying which task the sequence belongs to
        """
        bs, seq_len, _ = states.size()
        state_emb = self.state_embedding(states) # [bs, seq_len, d_model]
        task_emb = self.task_embedding(task_id) # [bs, d_model]
        inp = torch.cat([task_emb, state_emb], dim=1) # [bs, seq_len+1, d_model]
        inp = inp.transpose(0, 1)
        encoded = self.transformer(inp) # [seq_len+1, bs, d_model]
        encoded = encoded[1:] # [seq_len, bs, d_model]
        encoded = encoded.transpose(0, 1) # [bs, seq_len, d_model]
        actions_pred = self.action_head(encoded) # [bs, seq_len, action_dim]        
        return actions_pred
