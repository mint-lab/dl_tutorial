import torch
from transformers import BartConfig, BartForConditionalGeneration
from tf_translation_pytorch import encode


if __name__ == "__main__":
    # 1. Prepare vocabulary and training data
    # Note) P: padding, S: start token, E: end token
    tokens = [
        "P", "S", "E",
        "ich", "mochte", "ein", "bier",
        "i", "want", "a", "beer",
    ]
    vocab = {word: idx for idx, word in enumerate(tokens)}
    id_to_word = {idx: word for word, idx in vocab.items()}

    src = encode("ich mochte ein bier P", vocab)
    labels = encode("i want a beer E", vocab)

    # Hugging Face attention mask: 1 = valid, 0 = padding.
    attention_mask = (src != vocab["P"]).long()

    # Padding in labels must be replaced with -100 to ignore its loss.
    labels = labels.masked_fill(labels == vocab["P"], -100)

    # 2. Create a small BART initialized from scratch
    config = BartConfig(
        vocab_size=len(vocab),
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=32,
        activation_function="relu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=vocab["P"],
        bos_token_id=vocab["S"],
        eos_token_id=vocab["E"],
        decoder_start_token_id=vocab["S"],
        forced_eos_token_id=None,
    )
    model = BartForConditionalGeneration(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # 3. Training
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()

        # BART automatically creates shifted decoder inputs from labels.
        # It also applies the causal mask and computes cross-entropy loss.
        outputs = model(input_ids=src, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d}, loss: {loss.item():.4f}")

    # 4. Autoregressive inference
    model.eval()
    with torch.no_grad():
        generated = model.generate( input_ids=src, attention_mask=attention_mask, max_new_tokens=10, do_sample=False, num_beams=1)

    special_ids = {vocab["P"], vocab["S"], vocab["E"]}
    translation = " ".join(
        id_to_word[idx]
        for idx in generated[0].tolist()
        if idx not in special_ids
    )

    print("ich mochte ein bier ->", translation)