from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from llm_lifecycle_lab.model.native.attention import build_attention_mask


class NativeAttentionTests(unittest.TestCase):
    def test_unpadded_accelerator_prefill_uses_builtin_causal(self) -> None:
        for device in ("cuda", "mps"):
            with self.subTest(device=device):
                mask = build_attention_mask(
                    None,
                    query_length=4,
                    key_length=4,
                    past_length=0,
                    device=torch.device(device),
                )
                self.assertIsNone(mask)

    def test_cpu_prefill_uses_an_explicit_causal_mask(self) -> None:
        mask = build_attention_mask(
            None,
            query_length=4,
            key_length=4,
            past_length=0,
            device=torch.device("cpu"),
        )
        expected = torch.ones((4, 4), dtype=torch.bool).tril()[None, None]
        torch.testing.assert_close(mask, expected)

    def test_cached_mask_offsets_query_positions(self) -> None:
        mask = build_attention_mask(
            None,
            query_length=2,
            key_length=5,
            past_length=3,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        )[None, None]
        torch.testing.assert_close(mask, expected)

    def test_sdpa_causal_matches_explicit_outputs_and_gradients(self) -> None:
        torch.manual_seed(17)
        causal_inputs = [torch.randn(2, 4, 6, 8, requires_grad=True) for _ in range(3)]
        masked_inputs = [
            value.detach().clone().requires_grad_() for value in causal_inputs
        ]
        mask = torch.ones((6, 6), dtype=torch.bool).tril()
        causal = F.scaled_dot_product_attention(*causal_inputs, is_causal=True)
        explicit = F.scaled_dot_product_attention(*masked_inputs, attn_mask=mask)
        torch.testing.assert_close(causal, explicit)
        causal.square().sum().backward()
        explicit.square().sum().backward()
        for left, right in zip(causal_inputs, masked_inputs, strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
