import torch
from seq2seq_translation_tutorial import EncoderRNN, AttnDecoderRNN, evaluate, Lang


# Load checkpoint from seq2seq_mni_full.pth
checkpoint_path = "seq2seq_mni_full.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

hidden_size = checkpoint['hidden_size'] if 'hidden_size' in checkpoint else 128
MAX_LENGTH = checkpoint['max_length'] if 'max_length' in checkpoint else 128


if 'input_lang' in checkpoint and 'output_lang' in checkpoint:
    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']
else:
    print("❌ ERROR: input_lang and output_lang are missing from seq2seq_mni_full.pth. Cannot safely reconstruct vocabulary. Please retrain and save checkpoint with language objects included.")
    exit(1)

# Rebuild models

encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print(f"✅ Model loaded from {checkpoint_path}. Ready for translation!")

# Translate any sentence
while True:
    sentence = input("Enter English sentence (or 'quit' to exit): ")
    if sentence.strip().lower() == "quit":
        break
    output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    print("📝 Translation:", " ".join(output_words))
