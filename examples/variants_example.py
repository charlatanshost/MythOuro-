from mythouro import (
    mythouro_1b,
    MythOuro,
)

cfg = mythouro_1b()
model = MythOuro(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
