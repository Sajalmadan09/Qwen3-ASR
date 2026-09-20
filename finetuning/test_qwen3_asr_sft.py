import unittest
from types import SimpleNamespace

import torch
from torch import nn

from finetuning.qwen3_asr_sft import mask_prefix_labels, patch_outer_forward
from finetuning.qwen3_asr_multilingual_sft import MultilingualTrainer, language_weight_map


class DummyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(self, **kwargs):
        return kwargs


class DummyOuter(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = DummyThinker()


class PatchOuterForwardTests(unittest.TestCase):
    def test_delegates_embeddings_for_gradient_checkpointing(self):
        model = DummyOuter()
        patch_outer_forward(model)
        self.assertIs(model.get_input_embeddings(), model.thinker.embedding)
        replacement = nn.Embedding(8, 4)
        model.set_input_embeddings(replacement)
        self.assertIs(model.thinker.embedding, replacement)
        captured = model(input_ids=torch.tensor([[1]]), labels=torch.tensor([[1]]))
        self.assertIn("input_ids", captured)
        self.assertIn("labels", captured)

    def test_masks_prefix_after_left_padding(self):
        full = {
            "input_ids": torch.tensor([[0, 0, 10, 11, 20, 21], [10, 11, 12, 20, 21, 22]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        }
        prefix = {
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }
        labels = mask_prefix_labels(full, prefix, pad_token_id=0)
        self.assertEqual(labels.tolist()[0], [-100, -100, -100, -100, 20, 21])
        self.assertEqual(labels.tolist()[1], [-100, -100, -100, 20, 21, 22])

    def test_weighted_loss_emphasizes_selected_target_token(self):
        trainer = object.__new__(MultilingualTrainer)
        logits = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [3.0, 0.0]]])

        class DummyModel:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=logits)

        inputs = {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 0]]),
            "loss_weights": torch.tensor([[1.0, 4.0, 1.0]]),
        }
        loss = trainer.compute_loss(DummyModel(), inputs)
        first = torch.nn.functional.cross_entropy(logits[:, 0], torch.tensor([1]))
        second = torch.nn.functional.cross_entropy(logits[:, 1], torch.tensor([0]))
        self.assertTrue(torch.allclose(loss, (4 * first + second) / 5))

    def test_language_weight_parser(self):
        self.assertEqual(language_weight_map("English=2,Hindi=1.5"), {"english": 2.0, "hindi": 1.5})


if __name__ == "__main__":
    unittest.main()
