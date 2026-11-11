import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 1. Load the pretrained translation model and tokenizer.
model_name = "facebook/mbart-large-50-many-to-many-mmt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="de_DE")
model = MBartForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
model.eval()

# 2. Tokenize the German sentence.
text = "Ich möchte ein Bier."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 3. Generate an English translation.
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        max_new_tokens=40,
        num_beams=4,
        do_sample=False,
    )

# 4. Convert token IDs back to text.
translation = tokenizer.decode(generated[0], skip_special_tokens=True)

print(f"German : {text}")
print(f"English: {translation}")