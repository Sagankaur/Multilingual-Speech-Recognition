# Notebook comparison summary

## dlp-nppe-2 nb 1.ipynb
- Inference only, no training
- Model: `openai/whisper-base`
- Device forced to CPU
- Decoding: default `model.generate(...)` (greedy / no beam params)
- Cleaning: lowercase + remove punctuation
- Likely score around 1.0 because `whisper-base` is weaker and greedy decoding is less accurate

## nb cpu 0.8.ipynb
- Inference only, no training
- Model: `openai/whisper-small`
- Device forced to CPU in the strong path
- Decoding: `num_beams=5`, `max_new_tokens=255`
- Cleaning: lowercase + remove punctuation + collapse spaces
- Likely best leaderboard among the three because it is simple, zero-shot, and uses stronger decoding

## full-patched-notebook-gpt-pro penalities.ipynb
- Fine-tuning notebook
- Model: `openai/whisper-small` + LoRA-like patched wrapper
- Validation metric: raw local WER/CER with whitespace cleanup only
- Eval decode: `generation_num_beams=1`
- Training: 8 epochs, LR `1e-4`
- Final inference notebook cells add repetition penalties and `no_repeat_ngram_size=3`
- One final cleaner uppercases any text containing Latin letters, which can hurt WER if scoring is case-sensitive
- Its local validation WER is not directly comparable to the leaderboard WER from the simple inference notebooks

## Main reasons the numbers differ
1. Different model sizes: `base` vs `small`
2. Different decoding: greedy vs `num_beams=5`
3. Different metric strictness: raw local WER vs leaderboard submission score
4. Different text normalization: whitespace-only vs lowercase/punctuation stripping
5. Fine-tuning can overfit or drift across languages when the dataset is small and multilingual
6. CPU vs GPU itself is not the reason for WER differences if the code path is otherwise identical

## Practical recommendation
- Use `nb cpu 0.8.ipynb` as the current submission baseline.
- If fine-tuning, validate with the same decode settings you will use at test time.
- Remove the uppercase-for-English rule from the patched notebook.
- Track both raw WER and cleaned WER locally.
- Pick checkpoints by the metric that best matches the leaderboard, not by training loss alone.
