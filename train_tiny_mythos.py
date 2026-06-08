import torch
import torch.nn.functional as F
from mythouro.main import MythOuro, MythOuroConfig
from mythouro.tokenizer import MythOuroTokenizer

def main():
    print("Loading tokenizer...")
    try:
        tokenizer = MythOuroTokenizer("EleutherAI/gpt-neo-125m")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("\nSetting up a small MythOuro model...")
    cfg = MythOuroConfig(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=512,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=128
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MythOuro(cfg).to(device)

    # The text we want the model to memorize
    training_text = (
        "The quick brown fox jumps over the lazy dog. "
        "MythOuro is a Recurrent-Depth Transformer. "
        "It loops through the same parameters to reason deeper."
    )
    
    print(f"\nTarget text to memorize:\n'{training_text}'\n")
    
    # Encode and prepare input/target
    tokens = tokenizer.encode(training_text)
    x = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
    y = torch.tensor([tokens[1:]], dtype=torch.long).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 150
    
    print(f"Training for {epochs} iterations on {device}...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass: request 2 loops for training
        logits, unc = model(x, n_loops=2)
        
        # Calculate Cross Entropy Loss
        # logits shape: (B, T, vocab_size) -> (B*T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"Iteration {epoch:03d}/{epochs} | Loss: {loss.item():.4f}")

    print("\nTraining complete! Now let's test what it learned.\n")
    
    model.eval()
    
    # We provide the start of the sentence
    prompt = "The quick brown"
    print(f"Prompt: '{prompt}'")
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Generate the next tokens
        # We increase loops to 3 at inference time to show "depth extrapolation"
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=30, 
            n_loops=3, 
            temperature=0.1,  # Low temperature since we just want it to recite the memorized text
            top_k=1
        )
        
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens.tolist())
    
    print("\nModel Generation:")
    print("-" * 60)
    print(prompt + response)
    print("-" * 60)

if __name__ == "__main__":
    main()
