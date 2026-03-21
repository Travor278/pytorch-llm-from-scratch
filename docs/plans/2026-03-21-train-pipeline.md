# Train Pipeline Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CPU-friendly real-text training pipeline for the teaching Transformer, using a local TSV parallel corpus and a standalone `train.py` script.

**Architecture:** Keep the existing encoder-decoder model untouched and add a thin text-training layer around it. The new workflow will read TSV sentence pairs, build minimal vocabularies, convert text to padded tensors, create masks with the existing helpers, run teacher-forced training, evaluate on a validation split, and save the best checkpoint.

**Tech Stack:** Python, PyTorch, local TSV data, existing `Transformer/*.py` modules

---

### Task 1: Add a Small Real-Text Parallel Corpus

**Files:**
- Create: `Transformer/data/parallel_toy.tsv`

**Step 1: Create the dataset file**

Add 200 to 500 short source-target sentence pairs in TSV format:

```text
i like apples	我 喜欢 苹果
he likes tea	他 喜欢 茶
we read books	我们 读 书
```

**Step 2: Sanity-check formatting**

Rules:
- one pair per line
- source and target separated by a single tab
- tokenization already whitespace-friendly
- no blank target field

**Step 3: Manual verification**

Open the file and check:
- line count is in target range
- lines are short
- source/target look aligned

**Step 4: Commit**

```bash
git add Transformer/data/parallel_toy.tsv
git commit -m "data: add toy parallel corpus for transformer training"
```

### Task 2: Scaffold `train.py` Entry Structure

**Files:**
- Create: `Transformer/train.py`
- Reference: `Transformer/Transformer.py`
- Reference: `Transformer/create_mask.py`
- Reference: `Transformer/test.py`

**Step 1: Write a failing smoke path**

Create a `main()` that tries to:
- locate the dataset
- print a startup banner
- exit with an error if the dataset file is missing

Expected temporary behavior:
- script runs
- dataset path logic is explicit

**Step 2: Add imports and config container**

Define a simple config section with:
- data path
- batch size
- epochs
- model dimensions
- learning rate
- validation split
- checkpoint path

**Step 3: Add script entrypoint**

Use:

```python
if __name__ == "__main__":
    main()
```

**Step 4: Run script**

Run:

```bash
D:\Anaconda\python.exe Transformer\train.py
```

Expected:
- startup message
- no syntax errors

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: scaffold transformer training script"
```

### Task 3: Implement Dataset Loading and Vocabulary Building

**Files:**
- Modify: `Transformer/train.py`

**Step 1: Add TSV loading helpers**

Write functions to:
- read all lines
- split on tab
- strip whitespace
- skip malformed lines

Suggested signatures:

```python
def load_parallel_tsv(path: Path) -> list[tuple[str, str]]:
    ...
```

**Step 2: Add vocabulary building**

Build separate source and target vocabularies with:
- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

Suggested signatures:

```python
def build_vocab(texts: list[str]) -> dict[str, int]:
    ...
```

**Step 3: Add inverse vocabulary helper**

Needed for decoding predictions:

```python
def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    ...
```

**Step 4: Run script with debug prints**

Expected:
- pair count prints
- source/target vocab sizes print

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: add dataset loading and vocab building"
```

### Task 4: Encode Text and Build Padded Mini-Batches

**Files:**
- Modify: `Transformer/train.py`

**Step 1: Add text-to-id encoding**

Encode source text as token ids.
Encode target text with BOS/EOS handling.

Suggested signatures:

```python
def encode_source(text: str, vocab: dict[str, int]) -> list[int]:
    ...

def encode_target(text: str, vocab: dict[str, int]) -> list[int]:
    ...
```

**Step 2: Add train/validation split**

Use a deterministic split with a fixed random seed.

**Step 3: Add batch collation**

Implement padding per batch:

```python
def pad_sequences(seqs: list[list[int]], pad_idx: int) -> torch.Tensor:
    ...
```

and:

```python
def make_batches(examples: list[tuple[list[int], list[int]]], batch_size: int):
    ...
```

**Step 4: Add teacher forcing split**

Inside each batch:
- `tgt_input = tgt[:, :-1]`
- `tgt_output = tgt[:, 1:]`

**Step 5: Run and verify tensor shapes**

Expected:
- source batch `(B, S)`
- target batch `(B, T)`
- teacher-forcing tensors line up

**Step 6: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: add text encoding and padded batching"
```

### Task 5: Integrate Model, Masks, and Loss

**Files:**
- Modify: `Transformer/train.py`
- Reference: `Transformer/create_mask.py`
- Reference: `Transformer/Transformer.py`

**Step 1: Instantiate the model**

Use CPU-friendly defaults such as:

```python
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512,
    dropout=0.1,
    max_len=max_len,
)
```

**Step 2: Build masks per batch**

Use:
- `create_src_padding_mask`
- `create_tgt_mask`
- `create_memory_mask`

**Step 3: Add criterion and optimizer**

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

**Step 4: Run a single training step**

Expected:
- forward succeeds
- loss is finite
- backward succeeds

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: integrate transformer model into training loop"
```

### Task 6: Build Full Training and Validation Loops

**Files:**
- Modify: `Transformer/train.py`

**Step 1: Add `train_one_epoch`**

Responsibilities:
- `model.train()`
- iterate over training batches
- zero grad
- forward
- loss
- backward
- optimizer step
- accumulate average loss

**Step 2: Add `evaluate`**

Responsibilities:
- `model.eval()`
- `torch.no_grad()`
- forward only
- average validation loss

**Step 3: Add epoch loop**

Print:
- epoch number
- train loss
- val loss
- elapsed time

**Step 4: Run training**

Run:

```bash
D:\Anaconda\python.exe Transformer\train.py
```

Expected:
- several epochs complete
- losses remain finite
- train loss trends down

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: add training and validation loops"
```

### Task 7: Add Greedy Decoding Preview

**Files:**
- Modify: `Transformer/train.py`

**Step 1: Add a greedy decode helper**

Suggested signature:

```python
def greedy_decode(model, src_tensor, src_mask, tgt_vocab, max_len, bos_idx, eos_idx):
    ...
```

Use:
- encode once
- autoregressive decode loop
- `argmax` over generator logits
- stop on EOS

**Step 2: Add id-to-text reconstruction**

Convert decoded ids back to tokens using inverse vocab.

**Step 3: Print sample predictions**

After selected epochs, print:
- source text
- target text
- predicted text

**Step 4: Verify outputs are readable**

Expected:
- no crashes during decode
- predictions visibly improve after some training

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: add greedy decoding preview to training script"
```

### Task 8: Add Checkpoint Saving

**Files:**
- Modify: `Transformer/train.py`
- Create: `Transformer/checkpoints/` (runtime output directory)

**Step 1: Add checkpoint payload**

Save:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "src_vocab": src_vocab,
    "tgt_vocab": tgt_vocab,
    "config": config_dict,
    "best_val_loss": best_val_loss,
}
```

**Step 2: Save best checkpoint only**

Condition:
- new validation loss is lower than previous best

**Step 3: Print checkpoint path**

Expected:
- clear log line when best checkpoint updates

**Step 4: Verify checkpoint can be loaded**

Run a small smoke load with `torch.load(...)`.

**Step 5: Commit**

```bash
git add Transformer/train.py
git commit -m "feat: add checkpoint saving for transformer training"
```

### Task 9: Final Verification

**Files:**
- Verify: `Transformer/train.py`
- Verify: `Transformer/test.py`
- Verify: `Transformer/data/parallel_toy.tsv`

**Step 1: Run integration test**

```bash
D:\Anaconda\python.exe Transformer\test.py
```

Expected:
- all checks pass

**Step 2: Run training script**

```bash
D:\Anaconda\python.exe Transformer\train.py
```

Expected:
- training completes
- checkpoint saved
- sample predictions printed

**Step 3: Review runtime**

Check whether defaults are too slow or too fast on the user machine. Adjust:
- epochs
- model size
- dataset size
- batch size

while keeping the run educational and CPU-friendly.

**Step 4: Commit**

```bash
git add Transformer/train.py Transformer/data/parallel_toy.tsv
git commit -m "feat: add real-text transformer training pipeline"
```
