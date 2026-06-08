import os
import sys

# Force disable LIBUV before any PyTorch modules are loaded
os.environ["USE_LIBUV"] = "0"

from torch.distributed.run import main

if __name__ == "__main__":
    # Simulate the torchrun command line arguments
    sys.argv = ["torchrun", "--nproc_per_node=3", "training/1b_fine_web_edu.py"]
    print("Launching multi-GPU training with USE_LIBUV=0 enforced...")
    main()
