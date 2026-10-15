# Days 43-44 — Attention Is All You Need

## Problem
Days 41-42 produced attention as an isolated mechanism. A working
transformer needs it assembled into blocks with three other
components — residual connections, LayerNorm, and a position-wise
feed-forward network — and those blocks stacked into encoder and
decoder towers.

The decoder adds a third sublayer: cross-attention, where queries
come from the decoder but keys and values come from the encoder.

## Core Questions
- Why x + sublayer(x) rather than x = sublayer(x)?
- Why LayerNorm rather than BatchNorm?
- What does the feed-forward network do that attention cannot?
- Why apply positional encoding once rather than per block?
- Why must K and V come from the same source?
- Why does cross-attention never get a causal mask?

## Requirements
- Encoder block: self-attention + FFN, two residuals, two LayerNorms
- Decoder block: masked self-attention + cross-attention + FFN,
  three of each
- Stacks of N blocks with positional encoding applied once
- MultiHeadAttention generalized to two sources