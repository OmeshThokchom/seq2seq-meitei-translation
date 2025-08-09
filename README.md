# Seq2Seq English-Meitei Translation (PyTorch)

This project implements a sequence-to-sequence (Seq2Seq) neural translation model for English ↔ Meitei (Manipuri) using PyTorch. It is designed for large-scale training and inference, with support for attention and robust handling of unknown tokens.

🚀 Just pushed my UPDATED Seq2Seq translator for Meitei Mayek! 

Built on the **OFFICIAL PyTorch seq2seq with attention tutorial** (originally for French-English translation) :contentReference[oaicite:2]{index=2}.  
Added dope custom tokenizers, Meitei embeddings, and adapted the decoder logic to support our Mayek script.

Check it out: https://github.com/OmeshThokchom/seq2seq-meitei-translation  
Clone, train, test – and drop me your feedback or feature requests. Let’s make NLP lit for Meitei 💥


## Features
- Encoder-Decoder architecture with GRU layers
- Bahdanau Attention mechanism
- Handles unknown tokens (`<UNK>`) gracefully
- Efficient data loading and filtering for large datasets
- Training and inference scripts
- Checkpointing for model recovery and evaluation

## File Structure
- `seq2seq_translation_tutorial.py` — Main script for training, evaluation, and attention visualization
- `infrence.py` — Script for loading a trained model and running interactive inference
- `data/eng-mni-Mtei.txt` — Parallel corpus (English ↔ Meitei), tab-separated
- `checkpoints/` — Directory for saving model checkpoints

## Usage
### Training
1. Place your parallel corpus in `data/eng-mni-Mtei.txt` (tab-separated, one pair per line).
2. Adjust `MAX_LENGTH`, `batch_size`, and `hidden_size` in `seq2seq_translation_tutorial.py` to fit your hardware.
3. Run:
   ```bash
   python seq2seq_translation_tutorial.py
   ```
   - The script will filter malformed lines, truncate long sentences, and save checkpoints every few epochs.

### Inference
1. After training, run:
   ```bash
   python infrence.py
   ```
2. Enter English sentences interactively to get Meitei translations.
   - Unknown words will be mapped to `<UNK>`.
   - The model will load from `seq2seq_mni_full.pth`.

## Model Details
- **Encoder:** GRU-based, with dropout
- **Decoder:** GRU-based, with Bahdanau attention
- **Unknown tokens:** Any word not in the training vocabulary is mapped to `<UNK>`
- **EOS token:** All outputs are terminated with an EOS token

## Tips
- For large datasets, reduce `batch_size`, `hidden_size`, and `MAX_LENGTH` to avoid GPU out-of-memory errors.
- Only valid tab-separated lines are used for training.
- Checkpoints include model weights and vocabulary objects for safe inference.

## Requirements
- Python 3.8+
- PyTorch
- numpy
- matplotlib

## License
MIT
