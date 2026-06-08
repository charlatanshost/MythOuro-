"""
Tests for the SFT data path.

The single most important invariant is the loss mask: it MUST be zero on
prompt tokens and one on response tokens, with no off-by-one. A bug here
silently destroys SFT quality (the model wastes capacity predicting its
own prompts) without any visible training-time signal — losses still
look fine, but the model won't learn to respond.
"""

from __future__ import annotations

import torch

from mythouro.sft_data import (
    _build_sft_example,
    _to_messages_openhermes,
    _to_messages_magicoder,
    _to_messages_metamath,
)


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """
    Stand-in for `MythOuroTokenizer` that doesn't require downloading
    the real Ouro tokenizer at test time.

    Supports the surface `_build_sft_example` exercises:

      * `apply_chat_template(..., tokenize=True)`  → `list[int]`
      * `apply_chat_template(..., tokenize=False)` → a deterministic
        space-separated string of integer ids that `.encode()` parses
        back. This mirrors the production flow (render-as-text → encode)
        and lets the test assert that the round-trip preserves token
        counts.
      * `encode(text, add_special_tokens=False)`   → `list[int]`
      * `pad_token_id` / `eos_token_id`            attributes

    Token ids chosen for inspectability:

      * 100 = '<|im_start|>'
      * 101 = 'system' role
      * 102 = 'user' role
      * 103 = 'assistant' role
      * 104 = '<|im_end|>'
      * 105 = newline
      * Content tokens: hashed from the content string into [1000, 9999].
    """

    pad_token_id = 0
    eos_token_id = 104

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ):
        ids = self._build_ids(messages, add_generation_prompt)
        if tokenize:
            return ids
        # Render to a text form that round-trips through `.encode`.
        return " ".join(str(i) for i in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # `add_special_tokens` accepted for API parity with the real
        # MythOuroTokenizer; this fake doesn't model special tokens
        # because the chat template above already inserts them as
        # explicit ids (100/104/etc.), so adding more here would
        # double-count.
        del add_special_tokens
        if not text:
            return []
        return [int(tok) for tok in text.split()]

    def _build_ids(self, messages, add_generation_prompt: bool) -> list[int]:
        ids: list[int] = []
        role_map = {"system": 101, "user": 102, "assistant": 103}
        for m in messages:
            ids.append(100)                                 # <|im_start|>
            ids.append(role_map[m["role"]])                 # role
            ids.append(105)                                 # \n
            ids.extend(self._content_ids(m["content"]))
            ids.append(104)                                 # <|im_end|>
            ids.append(105)                                 # \n
        if add_generation_prompt:
            ids.append(100)                                 # <|im_start|>
            ids.append(103)                                 # assistant
            ids.append(105)                                 # \n
        return ids

    @staticmethod
    def _content_ids(content: str) -> list[int]:
        """Map each word to a deterministic token in [1000, 9999]."""
        return [1000 + (abs(hash(w)) % 9000) for w in content.split()]


# ---------------------------------------------------------------------------
# _build_sft_example — the core contract
# ---------------------------------------------------------------------------


class TestBuildSftExample:
    def setup_method(self):
        self.tok = _FakeTokenizer()
        self.seq_len = 64

    def _full_messages(self):
        return [
            {"role": "system",    "content": "be helpful"},
            {"role": "user",      "content": "what is two plus two"},
            {"role": "assistant", "content": "the answer is four"},
        ]

    # ── Shape & types ──

    def test_returns_three_tensors_of_seq_len(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        assert out is not None
        input_ids, target_ids, loss_mask = out
        assert input_ids.shape  == (self.seq_len,)
        assert target_ids.shape == (self.seq_len,)
        assert loss_mask.shape  == (self.seq_len,)
        assert input_ids.dtype  == torch.long
        assert target_ids.dtype == torch.long
        assert loss_mask.dtype  == torch.float32

    def test_input_target_are_shift_by_one(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        input_ids, target_ids, _ = out
        # In the un-padded region: target_ids[i] should equal input_ids[i+1]
        # because target is `tokens[1:]` and input is `tokens[:-1]`. The
        # equality only holds up to F-3 because at index F-2 the input
        # array has already started its padding while the target array
        # still has the final real token. That boundary mismatch is by
        # design — both arrays are padded independently *after* the
        # shift, not before.
        full_ids = self.tok.apply_chat_template(
            self._full_messages(),
            add_generation_prompt=False,
            tokenize=True,
        )
        F = len(full_ids)
        unpadded_overlap = F - 2
        assert torch.equal(
            target_ids[:unpadded_overlap],
            input_ids[1 : unpadded_overlap + 1],
        )

    # ── Loss mask: the critical invariant ──

    def test_loss_mask_is_zero_on_prompt_positions(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        _, _, loss_mask = out
        # Compute prompt length by re-rendering the prompt half. This
        # mirrors what `_build_sft_example` does internally, so if both
        # paths agree we know the implementation is consistent with the
        # contract we're testing.
        prompt_ids = self.tok.apply_chat_template(
            self._full_messages()[:-1],
            add_generation_prompt=True,
            tokenize=True,
        )
        P = len(prompt_ids)
        # Positions i < P-1 must be 0 (they predict prompt tokens).
        assert (loss_mask[: P - 1] == 0).all(), (
            "Loss mask leaks into prompt region — model would waste capacity "
            "learning to predict its own prompts."
        )

    def test_loss_mask_is_one_on_response_positions(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        _, _, loss_mask = out
        prompt_ids = self.tok.apply_chat_template(
            self._full_messages()[:-1],
            add_generation_prompt=True,
            tokenize=True,
        )
        full_ids = self.tok.apply_chat_template(
            self._full_messages(),
            add_generation_prompt=False,
            tokenize=True,
        )
        P = len(prompt_ids)
        F = len(full_ids)
        # Positions in [P-1, F-2] should be 1 (predicting response tokens
        # r_0 ... r_{R-1}). The -2 is because target_ids has length F-1.
        response_region_end = F - 1
        assert (loss_mask[P - 1 : response_region_end] == 1).all(), (
            "Loss mask missing on response region — gradient never reaches "
            "the tokens we want the model to learn."
        )

    def test_loss_mask_is_zero_in_padded_region(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        _, _, loss_mask = out
        full_ids = self.tok.apply_chat_template(
            self._full_messages(),
            add_generation_prompt=False,
            tokenize=True,
        )
        F = len(full_ids)
        # Beyond the original response, everything is padding.
        if F - 1 < self.seq_len:
            assert (loss_mask[F - 1 :] == 0).all()

    def test_loss_mask_sum_matches_response_token_count(self):
        out = _build_sft_example(self._full_messages(), self.tok, self.seq_len)
        _, _, loss_mask = out
        prompt_ids = self.tok.apply_chat_template(
            self._full_messages()[:-1],
            add_generation_prompt=True,
            tokenize=True,
        )
        full_ids = self.tok.apply_chat_template(
            self._full_messages(),
            add_generation_prompt=False,
            tokenize=True,
        )
        expected_response_tokens = len(full_ids) - len(prompt_ids)
        # When the response fits in seq_len, the mask sum should equal
        # the response token count exactly.
        if len(full_ids) <= self.seq_len + 1:
            assert int(loss_mask.sum().item()) == expected_response_tokens

    # ── Edge cases ──

    def test_returns_none_when_last_turn_is_not_assistant(self):
        # Prompt-only conversation — nothing to learn to predict.
        bad = [
            {"role": "user",   "content": "hello"},
            {"role": "system", "content": "be helpful"},  # last turn isn't assistant
        ]
        assert _build_sft_example(bad, self.tok, self.seq_len) is None

    def test_returns_none_when_empty_messages(self):
        assert _build_sft_example([], self.tok, self.seq_len) is None

    def test_returns_none_when_prompt_alone_exceeds_seq_len(self):
        # Single message containing thousands of words → prompt blows
        # past seq_len before any response token can land in the
        # loss-bearing region.
        huge = [
            {"role": "user",      "content": " ".join(["w"] * 5000)},
            {"role": "assistant", "content": "ok"},
        ]
        assert _build_sft_example(huge, self.tok, self.seq_len) is None

    def test_truncates_long_response_without_corrupting_mask(self):
        long_response = [
            {"role": "user",      "content": "give me a long story"},
            {"role": "assistant", "content": " ".join(["word"] * 1000)},
        ]
        out = _build_sft_example(long_response, self.tok, self.seq_len)
        assert out is not None
        input_ids, target_ids, loss_mask = out
        assert input_ids.shape == (self.seq_len,)
        # Most of the sequence is response now — mask should be
        # overwhelmingly 1 (not exactly all 1 because some prompt
        # tokens are at the start).
        assert loss_mask.mean().item() > 0.5


# ---------------------------------------------------------------------------
# Schema adapters
# ---------------------------------------------------------------------------


class TestSchemaAdapters:
    def test_openhermes_normaliser(self):
        sample = {
            "conversations": [
                {"from": "system", "value": "you are helpful"},
                {"from": "human",  "value": "what is 2+2"},
                {"from": "gpt",    "value": "4"},
            ]
        }
        msgs = _to_messages_openhermes(sample)
        assert msgs is not None
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[2]["content"] == "4"

    def test_openhermes_rejects_user_only(self):
        # Some OpenHermes rows are malformed and have no assistant turn.
        sample = {"conversations": [{"from": "human", "value": "hi"}]}
        assert _to_messages_openhermes(sample) is None

    def test_openhermes_rejects_empty(self):
        assert _to_messages_openhermes({"conversations": []}) is None
        assert _to_messages_openhermes({}) is None

    def test_magicoder_normaliser(self):
        sample = {
            "instruction": "write fib",
            "response":    "def fib(n): ...",
        }
        msgs = _to_messages_magicoder(sample)
        assert msgs is not None
        assert msgs == [
            {"role": "user",      "content": "write fib"},
            {"role": "assistant", "content": "def fib(n): ..."},
        ]

    def test_magicoder_rejects_missing_field(self):
        assert _to_messages_magicoder({"instruction": "x"}) is None
        assert _to_messages_magicoder({"response": "x"}) is None

    def test_metamath_normaliser(self):
        sample = {
            "query":    "what is 2+2",
            "response": "step 1: ... so the answer is 4",
        }
        msgs = _to_messages_metamath(sample)
        assert msgs is not None
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_metamath_rejects_missing_field(self):
        assert _to_messages_metamath({"query": "x"}) is None


# ---------------------------------------------------------------------------
# Masked CE — end-to-end loss correctness
# ---------------------------------------------------------------------------


class TestMaskedCeLoss:
    """
    Verify the masked CE formula in training/sft.py behaves correctly
    on hand-constructed cases. The implementation lives in training/sft.py
    so we import it explicitly.
    """

    def setup_method(self):
        from training.sft import masked_ce_loss
        self.masked_ce_loss = masked_ce_loss

    def test_all_masked_returns_zero(self):
        # When loss_mask is all zeros, the loss should be 0 — not NaN
        # from dividing by zero. The `.clamp(min=1.0)` guard handles this.
        B, T, V = 2, 8, 50
        logits  = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask    = torch.zeros(B, T)
        loss = self.masked_ce_loss(logits, targets, mask)
        assert loss.item() == 0.0

    def test_all_unmasked_matches_plain_ce(self):
        # When loss_mask is all ones, masked CE should equal F.cross_entropy
        # with reduction="mean" (modulo floating point).
        import torch.nn.functional as F
        B, T, V = 2, 8, 50
        torch.manual_seed(0)
        logits  = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask    = torch.ones(B, T)

        masked = self.masked_ce_loss(logits, targets, mask)
        plain  = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        assert torch.allclose(masked, plain, atol=1e-6)

    def test_half_masked_averages_only_unmasked(self):
        # Mask the first half of positions. The result should equal
        # plain CE computed only on the second half.
        import torch.nn.functional as F
        B, T, V = 1, 10, 50
        torch.manual_seed(0)
        logits  = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask    = torch.zeros(B, T)
        mask[:, 5:] = 1.0

        masked = self.masked_ce_loss(logits, targets, mask)
        plain_second_half = F.cross_entropy(
            logits[:, 5:].reshape(-1, V),
            targets[:, 5:].reshape(-1),
            reduction="mean",
        )
        assert torch.allclose(masked, plain_second_half, atol=1e-6)
