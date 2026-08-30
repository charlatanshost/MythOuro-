import torch
from mythouro.main import MythOuro, MythOuroConfig
from mythouro.tokenizer import MythOuroTokenizer
from mythouro.inference import uncertainty_gated_generate

def main():
    print("Loading tokenizer...")
    try:
        # Defaults to "openai/gpt-oss-20b" or "EleutherAI/gpt-neox-20b"
        # Using a fast alternative if not already cached
        tokenizer = MythOuroTokenizer("EleutherAI/gpt-neo-125m")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("\nSetting up a small MythOuro model (untrained, random weights)...")
    # We use a very small config so it runs quickly on any machine
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
    
    model = MythOuro(cfg)
    model.eval()
    print("Model ready!\n")
    print("=" * 60)
    print("Note: This model has random, untrained weights.")
    print("It will output gibberish, but this demonstrates the")
    print("full inference pipeline with ACT halting and CoT loops.")
    print("=" * 60)

    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.strip().lower() in ["quit", "exit"]:
            break
        if not prompt.strip():
            continue

        print("\nGenerating...")
        
        # 1. Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        
        # 2. Generate with our new Phase 2C Uncertainty Gated Generator
        with torch.no_grad():
            output_ids = uncertainty_gated_generate(
                model, 
                input_ids,
                max_new_tokens=20,
                min_loops=1,
                max_loops=cfg.max_loop_iters,
                threshold=0.5,
                temperature=1.0,
                top_k=50
            )
            
        # 3. Decode output
        # Get just the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens.tolist())
        
        print("\nResponse:")
        print(response)
        print("-" * 60)

if __name__ == "__main__":
    main()
