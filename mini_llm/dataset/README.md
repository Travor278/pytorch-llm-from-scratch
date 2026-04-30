# mini_llm dataset

Keep local training files here. Actual data is ignored by git.

Expected formats:

Pretrain:

```json
{"text": "Transformer uses self-attention to model context."}
```

SFT:

```json
{"conversations": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
```

DPO:

```json
{"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good"}], "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad"}]}
```
