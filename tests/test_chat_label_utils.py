import torch

from starVLA.model.modules.vlm.chat_label_utils import IGNORE_INDEX, mask_labels_to_response


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "prompt": [10, 11],
            "answer": [20, 21],
        }
        return {"input_ids": mapping[text]}


def test_mask_labels_to_response_matches_response_after_expanded_image_tokens():
    batch_inputs = {
        "input_ids": torch.tensor([[10, 99, 99, 99, 11, 20, 21, 30]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
    }

    labels = mask_labels_to_response(
        batch_inputs,
        _FakeTokenizer(),
        prompt_texts=["prompt"],
        response_texts=["answer"],
    )

    assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, 30]]

