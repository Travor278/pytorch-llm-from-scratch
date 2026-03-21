# Train Pipeline Design

**Topic:** Real-text CPU-friendly training pipeline for the teaching Transformer implementation

**Date:** 2026-03-21

## Goal

Add a `train.py` workflow that trains the existing encoder-decoder Transformer on a small real-text parallel corpus using CPU-friendly defaults, while keeping the code understandable for learning.

## Context

The project already contains:

- `Transformer/PostionalEncoding.py`
- `Transformer/MHA.py`
- `Transformer/FFN.py`
- `Transformer/Encoder.py`
- `Transformer/Decoder.py`
- `Transformer/create_mask.py`
- `Transformer/Transformer.py`
- `Transformer/test.py`

These files cover the model architecture and integration checks. What is missing is a realistic training script that:

- reads text instead of hard-coded tensors
- builds vocabularies
- creates padded batches
- applies teacher forcing
- trains with masks and loss
- saves checkpoints
- prints decoded predictions

## Chosen Direction

Use a local text dataset stored as a TSV file:

- path: `Transformer/data/parallel_toy.tsv`
- format: one pair per line, `source<TAB>target`

Example:

```text
i like apples	我 喜欢 苹果
he likes tea	他 喜欢 茶
```

This keeps the project close to a real seq2seq workflow without introducing external tokenizer libraries or heavyweight dataset tooling.

## Why This Direction

Compared with synthetic numeric data, this better demonstrates how a text training pipeline really works:

- text -> tokenization -> ids -> padding -> masks -> model -> loss

Compared with a full production tokenizer/data stack, this stays small enough to understand and run on CPU.

## Data Design

The dataset will be:

- real short text pairs
- intentionally small
- large enough to learn visible mappings

Target scale:

- roughly 200 to 500 source-target pairs
- sentence length mostly 2 to 8 tokens

The initial implementation will use whitespace tokenization. This is intentionally simple and teaching-oriented.

Special tokens:

- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

Separate vocabularies will be built for source and target text.

## Training Pipeline Design

`Transformer/train.py` will include:

1. dataset loading from TSV
2. train/validation split
3. vocabulary construction
4. text-to-id encoding
5. mini-batch collation with padding
6. target shifting for teacher forcing
7. mask creation using `create_mask.py`
8. forward pass through `Transformer.py`
9. loss computation with `CrossEntropyLoss(ignore_index=pad_idx)`
10. backward pass and optimizer update
11. validation loop
12. greedy decoding preview
13. checkpoint saving

The script should be runnable as a standalone teaching example.

## Model and Runtime Constraints

The default configuration should be CPU-friendly:

- `d_model` around `96` or `128`
- `num_heads = 4`
- `num_encoder_layers = 2`
- `num_decoder_layers = 2`
- `d_ff` around `256` or `512`
- moderate batch size such as `16`

The goal is not SOTA quality. The goal is:

- stable CPU execution
- visible learning progress
- a training run that can reasonably fit into about half an hour on the user’s machine

The final runtime can be controlled by:

- dataset size
- batch size
- epoch count
- maximum sentence length

## Outputs

The training workflow should produce:

- console logs for train and validation loss
- decoded sample predictions during training
- best checkpoint saved under `Transformer/checkpoints/`
- saved vocabularies and hyperparameters inside the checkpoint payload

Suggested checkpoint payload:

- `model_state_dict`
- `optimizer_state_dict`
- `src_vocab`
- `tgt_vocab`
- training hyperparameters
- best validation loss

## Success Criteria

The implementation is successful if:

1. `train.py` runs end-to-end on CPU
2. training loss decreases clearly
3. validation loss is finite and generally trends down
4. greedy decoding shows partially or clearly correct predictions on held-out examples
5. checkpoints save correctly
6. no mask, shape, or NaN issues appear during a normal run

## Non-Goals

The first version will not include:

- external tokenizer libraries
- beam search
- BLEU or advanced metrics
- mixed precision
- GPU-specific optimization
- distributed training
- resume-from-checkpoint support

Those can be added later if needed, but they are not required for the first educational training pipeline.
