# Days 43-44 — Attention Is All You Need

Full encoder-decoder transformer from scratch.

## Components
PositionEncoding      — fixed sinusoidal (Day 40)
MultiHeadAttention    — two-source: forward(query_src, kv_src, mask)
SingleEncoder         — self-attn + FFN, 2 residuals, 2 LayerNorms
Encoder               — N blocks, positional encoding applied once
SingleDecoder         — masked self-attn + cross-attn + FFN, 3 each
Decoder               — N blocks

## Shapes
    encoder: (2, 10, 128) → (2, 10, 128)
    decoder: xe (2, 10, 128), xd (2, 6, 128) → (2, 6, 128)

Output takes the TARGET length. Cross-attention scores are
rectangular: (batch, heads, s_q, s_kv).

## Parameters (d=128, heads=8, 6 blocks)
| Stack   | Parameters |
|---------|-----------:|
| Encoder |  1,186,560 |
| Decoder |  1,581,312 |

FFN is ~67% of each block.

## Key Insights
Residual = gradient multiplied by 1 across depth. Same as the LSTM
cell state across time.

Attention routes between tokens. The FFN processes each token in
isolation and holds most of the parameters. Routing is cheap,
storage is expensive.

Cross-attention: Q from decoder, K AND V from encoder, never split,
never causally masked.

## Run
python encoder.py
python decoder.py