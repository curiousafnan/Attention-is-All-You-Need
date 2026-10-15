import torch
import math
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,seq_len:int,hidden_state:int)->torch.tensor:
        pos = torch.zeros(seq_len,hidden_state)
        max_pos = torch.arange(seq_len).unsqueeze(1)
        fun = torch.exp(torch.arange(0, hidden_state, 2).float() * (-math.log(10000.0) / hidden_state))
        angle = max_pos * fun
        pos[:,0::2] = torch.sin(angle)
        pos[:,1::2] = torch.cos(angle)
        return pos


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_states,no_of_heads):
        super().__init__()
        self.hidden = hidden_states
        self.heads = no_of_heads
        self.head_dim = hidden_states // no_of_heads
        self.q = nn.Linear(self.hidden,self.hidden,bias=False)
        self.k = nn.Linear(self.hidden,self.hidden,bias=False)
        self.v = nn.Linear(self.hidden,self.hidden,bias=False)
        self.w = nn.Linear(self.hidden,self.hidden,bias=False)

    def forward(self,vector,mask=None):
        batch,seq_len,_=vector.shape
        q = self.q(vector)
        k = self.k(vector)
        v = self.v(vector)
        q_h = q.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        k_h = k.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        v_h = v.reshape(batch,seq_len,self.heads,-1).transpose(1,2)
        k_t = torch.transpose(k_h,-2,-1)
        scores = (q_h@k_t) / math.sqrt(self.head_dim)
        if mask is not None:
            mask_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask_matrix,float('-inf'))
        pred_q_k = torch.softmax(scores,dim=-1)
        output = (pred_q_k @ v_h)
        output = torch.transpose(output,1,2).reshape(batch,seq_len,-1)
        final_output = self.w(output)
        return final_output , pred_q_k


class SingleEncoder(nn.Module):
    def __init__(self, hidden_states,no_of_heads=8):
        super().__init__()
        self.hidden_states = hidden_states
        self.no_of_heads = no_of_heads
        self.norm1 = nn.LayerNorm(hidden_states)
        self.norm2 = nn.LayerNorm(hidden_states)
        self.multi_head = MultiHeadAttention(hidden_states=self.hidden_states,no_of_heads=self.no_of_heads)
        self.l1 = nn.Linear(hidden_states,4*hidden_states,bias=True)
        self.l2 = nn.Linear(4*hidden_states,hidden_states,bias=True)

    def forward(self,x:torch.tensor,mask=None):
        context_embedding,_ = self.multi_head(x,mask=mask)
        z = x + context_embedding
        z_dash = self.norm1(z)
        output = self.l2(torch.relu(self.l1(z_dash)))
        final_output = self.norm2(output+z_dash)
        return final_output


class Encoder(nn.Module):
    def __init__(self, no_of_components,hidden_states, no_of_heads,mask=None):
        super().__init__()
        self.no_of_components = no_of_components
        self.hidden_states = hidden_states
        self.no_of_heads = no_of_heads
        self.mask = mask
        self.pos_enc = PositionEncoding()
        self.encoders = nn.ModuleList()
        for _ in range(self.no_of_components):
            self.encoders.append(SingleEncoder(self.hidden_states,no_of_heads))

    def forward(self,x: torch.tensor):
        _,seq_len,_ = x.shape
        positional_encoding = self.pos_enc(seq_len,self.hidden_states)
        x = x+positional_encoding
        for encoder in self.encoders:
            x = encoder(x,mask=self.mask)
        return x

if __name__ == "__main__":
    en = Encoder(no_of_components=6, hidden_states=128, no_of_heads=8)
    x = torch.randn(2, 6, 128)
    print(en(x))
    print(sum(p.numel() for p in en.parameters()))
