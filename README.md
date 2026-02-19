# LLM-Lite

Clean and minimal LLM training / inference code, based on the Transformer (LLaMA) architecture.

## Features

- Pure PyTorch implementation of the LLaMA-family Transformer
- KV-cache for efficient autoregressive generation
- Speculative decoding support (draft model + target model)
- Tensor parallelism for multi-GPU inference
- Tiktoken (Llama-3) and SentencePiece tokenizer support
- Integration with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- HuggingFace checkpoint conversion utility

## Supported Models

| Model | Config Key |
|-------|-----------|
| LLaMA-2 7B | `7B` |
| LLaMA-2 13B | `13B` |
| LLaMA-2 70B | `70B` |
| LLaMA-3 8B | `Llama-3-8B` |
| Mistral 7B | `Mistral-7B` |
| CodeLlama 7B | `CodeLlama-7b-Python-hf` |
| CodeLlama 34B | `34B` |

## Quick Start

### 1. Convert Checkpoint

```bash
python scripts/convert_ft_ckpt.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```

### 2. Generate Text

```bash
python generate.py --prompt "Hello, my name is" \
    --checkpoint_path checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth \
    --max_new_tokens 200
```

### 3. Evaluate

```bash
python eval.py --checkpoint_path checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth \
    --tasks hellaswag
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.2
- sentencepiece
- tiktoken

## License

See [LICENSE](LICENSE) for details.
