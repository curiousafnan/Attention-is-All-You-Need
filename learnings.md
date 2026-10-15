# Learnings — Days 43-44

## Residual Connections: x + sublayer(x)
NOT recurrent connections — recurrence is the RNN/LSTM mechanism
removed on Day 41. These are residual (or skip) connections.

Without them, each layer's output replaces its input entirely.
Gradient travelling backward from block 12 to block 1 passes through
every layer's weight matrices and shrinks toward zero. Early layers
stop learning.

With the addition, the gradient has a path multiplied by 1 at each
block and arrives intact.

This is exactly the Day 38 derivation — ∂c_t/∂c_{t-1} = 1 for an
addition — applied to DEPTH instead of TIME:

    RNN:          h_t = tanh(...)        replacement across time
    LSTM:         c_t = f⊙c + i⊙g        addition across time
    Plain stack:  x = layer(x)           replacement across depth
    Transformer:  x = x + layer(x)       addition across depth

Second benefit: the sublayer only learns a CORRECTION to x, not a
replacement. A block with nothing useful to contribute can learn to
output near-zero and let x pass through unchanged. Stacking 96
layers is only possible because each can be nearly a no-op.

## LayerNorm, Not BatchNorm
BatchNorm normalizes across the batch — statistics computed from
other examples. LayerNorm normalizes across the FEATURE dimension,
per token, independently of everything else in the batch.

Why it matters for sequences: batches contain sentences of unequal
length, padded with zeros. BatchNorm's statistics would be polluted
by that padding. LayerNorm never looks across examples, so padding
cannot contaminate it.

Distinct from the /√d_k in attention: that is a fixed constant
known before seeing data, correcting for how many terms a dot
product sums. LayerNorm computes mean and std FROM the data and has
learnable γ and β.

Visible in the output: values come back to roughly [-2, 2] after
every block regardless of what attention and the FFN did to the
magnitudes.

## The FFN Is ~67% of the Parameters
    attention (Q,K,V,W_o):  65,536   33%
    FFN (two Linears):     131,712   67%

This holds in real models — GPT-3's FFN layers are roughly
two-thirds of its parameters.

What each does:
    ATTENTION is ROUTING. It decides which information moves where
    — which tokens are relevant to which. All inter-token movement
    happens here.

    FFN is STORAGE AND PROCESSING. Two Linear layers applied
    independently to each token position. It never looks at any
    other token.

That second point is the one to hold onto: the FFN cannot do
translation, agreement, or anything involving relationships between
words. It transforms each position in isolation. Current research
suggests it holds much of the model's factual knowledge.

Routing is cheap. Storage is expensive. That is the split.

## Positional Encoding Applied Once
Not because "the sentence doesn't change" — because each block
PRESERVES the position information. Residual connections mean x
passes through intact, so a position signal added at the bottom is
still present at block 6.

Adding it per block would stack six copies of the same signal on
top of accumulating representations, strengthening position
relative to content with every layer, for no reason.

## K and V Must Share a Source
In cross-attention, Q comes from the decoder and K,V from the
encoder. K and V can never be split across the two.

Mechanically: α is (batch, s_q, s_kv), indexed by SOURCE position
in its last dimension. The thing being blended must therefore also
be indexed by source position. Splitting them produces a shape
error — caught this on paper before coding.

Semantically: K is what makes a source position findable, V is what
gets taken from it. Both describe the same position. They are two
views of one thing, not two different things.

## Cross-Attention Is Never Causally Masked
Causal masking exists so a position cannot see its own answer
during parallel training. In cross-attention there is no such
risk — the decoder is always allowed to see the ENTIRE source. The
source is input, not something being predicted.

Self-attention in the same block IS masked, because there the
decoder is attending to its own output, where future positions are
the answers.

So within one decoder block: mask on sublayer 1, no mask on
sublayer 2.

## BERT vs GPT vs the Original
    Original (2017):  encoder + decoder, causal mask in decoder only
    BERT:             encoder only, NO causal mask, cannot generate
    GPT:              decoder only, causal mask always, no cross-attn

GPT's "decoder" blocks are structurally encoder blocks with the
mask turned on — the cross-attention sublayer is dropped since
there is no encoder.

In this code the entire BERT/GPT distinction is mask=None versus
mask=True.

## Bugs Fixed
- torch.norm used in place of LayerNorm. torch.norm computes a
  vector's MAGNITUDE and collapses the dimension; LayerNorm is
  (x-mean)/std and preserves shape
- torch.layer_norm(z, dim=1) — wrong function and signature; the
  module nn.LayerNorm(d) is called with just the tensor
- mask=False passed where mask=None was meant — the check is
  `if mask is not None`, and False is not None, so the mask applied
- ReLU on the FFN's second Linear as well as the first
- Hardcoded 2048 instead of 4 × d_model
- Positional encoding applied inside each block instead of once
- PositionEncoding built on batch size instead of sequence length
- norm2 reused for the decoder's third sublayer; norm3 declared
  but unused
- Causal mask passed to cross-attention and hardcoded True on
  self-attention — exactly backwards
- Mask built as torch.ones(batch, seq) — the mask is square over
  sequence positions and has nothing to do with batch
- Decoder.forward referenced undefined `x` and `self.encoders`,
  and did not thread the loop output back into xd