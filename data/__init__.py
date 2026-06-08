"""
MythOuro data-pipeline utilities — preprocessing that runs entirely on CPU
before any GPU touches the dataset.

Three primitives, each with a CLI:

    data/dedup.py            Near-duplicate removal via MinHash LSH
    data/contamination.py    n-gram overlap filter vs eval benchmarks
    data/tokenizer_eval.py   Comparative tokenizer compression analysis

Use cases
---------
Before launching a real training run, the recommended order is:
    1. Tokenizer eval → pick the tokenizer that compresses your domain best.
       Locks the tokenizer choice for the rest of the project.
    2. Contamination filter → strip any training docs that overlap ARC /
       GSM8K / HumanEval test prompts. Without this your eval numbers
       are meaningless.
    3. Dedup → MinHash LSH on the surviving corpus. Cheap, large quality
       lift, especially on web data with crawl-time duplication.

All three are designed to be runnable on the user's Xeon 8480 (60 cores,
no GPU required) while training infrastructure is being finalised.
"""

from data.dedup import MinHashDeduplicator
from data.contamination import ContaminationFilter
from data.tokenizer_eval import compare_tokenizers

__all__ = [
    "MinHashDeduplicator",
    "ContaminationFilter",
    "compare_tokenizers",
]
