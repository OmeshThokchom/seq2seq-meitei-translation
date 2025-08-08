import torch
from seq2seq_translation_tutorial import EncoderRNN, AttnDecoderRNN, evaluate

# Load checkpoint
checkpoint = torch.load("seq2seq_mni_full.pth", map_location="cpu")

hidden_size = checkpoint['hidden_size']
MAX_LENGTH = checkpoint['max_length']
input_lang = checkpoint['input_lang']
output_lang = checkpoint['output_lang']

# Rebuild models
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print("✅ Model loaded and ready for inference!")

# Example inference
output_words, attentions = evaluate(encoder, decoder, "hello world", input_lang, output_lang)
print("Translated:", " ".join(output_words))
import torch
from seq2seq_translation_tutorial import EncoderRNN, AttnDecoderRNN, evaluate

# 1️⃣ Load checkpoint
checkpoint = torch.load("seq2seq_mni_full.pth", map_location="cpu")

hidden_size = checkpoint['hidden_size']
MAX_LENGTH = checkpoint['max_length']
input_lang = checkpoint['input_lang']
output_lang = checkpoint['output_lang']

# 2️⃣ Rebuild models
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

print("✅ Model loaded. Ready for translation!")

# 3️⃣ Translate any sentence
while True:
    sentence = input("Enter English sentence: ")
    if sentence.strip().lower() == "quit":
        break
    output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang)
    print("📝 Translation:", " ".join(output_words))
