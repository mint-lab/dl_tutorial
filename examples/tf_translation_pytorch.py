import torch
import torch.nn as nn

# Transformer model
class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=32, max_len=32):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # src, tgt: [batch, sequence_length]
        src_pos = torch.arange(src.size(1), device=src.device)
        tgt_pos = torch.arange(tgt.size(1), device=tgt.device)

        src_emb = self.src_emb(src) + self.pos_emb(src_pos)
        tgt_emb = self.tgt_emb(tgt) + self.pos_emb(tgt_pos)

        # True means attention is blocked.
        # Block future target tokens to prevent information leakage.
        causal_mask = torch.triu(
            torch.ones(
                tgt.size(1), tgt.size(1),
                dtype=torch.bool, device=tgt.device,
            ),
            diagonal=1,
        )

        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=(src == 0),
            tgt_key_padding_mask=(tgt == 0),
            memory_key_padding_mask=(src == 0),
        )
        return self.output(hidden)  # [batch, target_length, target_vocab_size]


def encode(text, vocab):
    return torch.tensor([[vocab[word] for word in text.split()]])


if __name__ == "__main__":
    # 1. Prepare vocabulary and training data
    # Note) P: padding, S: start token, E: end token
    src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4}
    tgt_vocab = {"P": 0, "i": 1, "want": 2, "a": 3, "beer": 4, "S": 5, "E": 6}
    id_to_word = {idx: word for word, idx in tgt_vocab.items()}

    src = encode("ich mochte ein bier P", src_vocab)  # [1, 5]
    tgt_in = encode("S i want a beer", tgt_vocab)     # [1, 5]
    tgt_out = encode("i want a beer E", tgt_vocab)    # [1, 5]

    # 2. Create the model
    model = Translator(len(src_vocab), len(tgt_vocab))
    # Disable the experimental nested-tensor inference optimization.
    model.transformer.encoder.use_nested_tensor = False
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 3. Training with teacher forcing
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d}, loss: {loss.item():.4f}")

    # 4. Autoregressive inference: start with S and generate until E
    model.eval()
    with torch.no_grad():
        generated = torch.tensor([[tgt_vocab["S"]]])
        words = []

        for _ in range(10):
            logits = model(src, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            if next_token.item() == tgt_vocab["E"]:
                break

            words.append(id_to_word[next_token.item()])
            generated = torch.cat([generated, next_token], dim=1)

    print("ich mochte ein bier ->", " ".join(words))