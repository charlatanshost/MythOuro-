"""
HuggingFace tokenizer wrapper for MythOuro.

Default tokenizer
-----------------
`ByteDance/Ouro-2.6B-Thinking`. This is the Ouro looped-LM variant we
distil from (see `training/distill.py`), and aligning the student's
vocabulary to the teacher's is a precondition for logit-level
distillation. Picking it as the default makes the common
"MythOuro + Ouro distillation" path zero-config.

Properties of the default tokenizer:
    * GPT2Tokenizer (BPE), vocab_size = 49152
    * Special tokens (id):
        0  <|endoftext|>
        1  <|im_start|>      (BOS)
        2  <|im_end|>        (EOS)
        3  <think>
        4  </think>
        5  <file_sep>
    * Chat template: ChatML (`<|im_start|>role\\n…<|im_end|>\\n`), with
      optional `<think>…</think>` reasoning blocks.

To switch tokenizers for an experiment, pass `model_id=`; nothing in
the library hard-codes the default beyond the constant below.
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"


class MythOuroTokenizer:
    """
    Minimal-surface tokenizer wrapper. The underlying HF tokenizer is
    exposed as `.tokenizer` for any operation we don't proxy through
    explicitly.

    Args:
        model_id (str): The HuggingFace model id or local path. Defaults
            to `DEFAULT_MODEL_ID` (Ouro). Pass another id (e.g.
            "openai-community/gpt2") for ablation runs.

    Example:
        >>> tok = MythOuroTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
        >>> tok.vocab_size
        49152
        >>> tok.eos_token_id
        2
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id

    # ------------------------------------------------------------------
    # Core properties / passthroughs
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return self.tokenizer.vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning-of-sequence token id (None if undefined)."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """End-of-sequence token id (None if undefined)."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """Padding token id (None if undefined)."""
        return self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode `text` to token ids.

        `add_special_tokens=False` by default — most MythOuro training
        paths pack text into fixed-length chunks where adding BOS/EOS
        per call would distort the token count and waste budget. Pass
        `True` explicitly for chat-template or SFT use cases that need
        the framing tokens.
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode `token_ids` back to text. Special tokens stripped by default."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Chat-template support (Ouro uses ChatML)
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tokenize: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Apply the underlying tokenizer's chat template to a message list.

        For Ouro (the default) this renders ChatML with optional
        `<think>` reasoning blocks. Set `enable_thinking=True` to
        prepend `<think>\\n` to the assistant turn — Ouro is trained
        to emit reasoning inside that block when prompted.

        Args:
            messages              : list of `{"role": ..., "content": ...}` dicts.
            add_generation_prompt : when True, append the `<|im_start|>assistant`
                                    prefix so the model knows to start
                                    generating its turn.
            enable_thinking       : when True (Ouro-specific), open a
                                    `<think>` block after the assistant prefix.
            tokenize              : when False, returns the rendered string
                                    instead of token ids.
            return_tensors        : "pt" / "np" / None — forwarded to HF.

        Falls back gracefully if the underlying tokenizer has no chat
        template (e.g. raw GPT-2 BPE), raising a clear message.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise NotImplementedError(
                f"Tokenizer {self.model_id!r} has no apply_chat_template; "
                "use a chat-template-aware tokenizer (default Ouro) or "
                "format messages manually."
            )
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tokenize=tokenize,
            return_tensors=return_tensors,
            **kwargs,
        )
