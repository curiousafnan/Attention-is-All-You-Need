# Approach — Days 43-44

## Encoder Block
    attn, _ = self_attention(x, x, mask=mask)
    x = norm1(x + attn)
    ff  = l2(relu(l1(x)))
    x = norm2(x + ff)

Two sublayers, each wrapped in residual + LayerNorm.

## Decoder Block
    attn1, _ = self_attention(xd, xd, mask=mask)      causal
    xd = norm1(xd + attn1)
    attn2, _ = cross_attention(xd, xe, mask=None)     never masked
    xd = norm2(xd + attn2)
    ff = l2(relu(l1(xd)))
    xd = norm3(xd + ff)

Three sublayers, three residuals, three LayerNorms.

## Generalizing MultiHeadAttention to Two Sources
    forward(query_source, kv_source, mask=None)

Self-attention passes the same tensor twice; cross-attention passes
different ones. No mode flag, no branch — the call site reads as
what it does.

Two lengths must be tracked separately:
    q  from query_source   (batch, s_q, h)
    k  from kv_source      (batch, s_kv, h)
    v  from kv_source      (batch, s_kv, h)

    q @ kᵀ  →  (batch, heads, s_q, s_kv)    rectangular
    α @ v   →  (batch, heads, s_q, head_dim)

For translation s_q ≠ s_kv — that asymmetry is the whole reason
seq2seq exists.

## Feed-Forward Network
    Linear(d, 4d) → ReLU → Linear(4d, d)

One activation, in the middle. The 4× inner width is convention.

## Parameters (d=128, heads=8, 6 blocks)
    Encoder block:
      attention (Q,K,V,W_o)  4 × 128 × 128    =  65,536
      FFN l1 (128→512)       128×512 + 512    =  66,048
      FFN l2 (512→128)       512×128 + 128    =  65,664
      LayerNorm × 2          2 × 2 × 128      =     512
                                                 197,760
      × 6                                      1,186,560

    Decoder stack (adds cross-attention + norm per block):
                                               1,581,312

    Difference: 394,752 = 6 × 65,792, exactly the extra
    cross-attention block plus its LayerNorm.