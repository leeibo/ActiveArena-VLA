import torch

IGNORE_INDEX = -100


def _prompt_token_length(tokenizer, prompt_text: str) -> int:
    tokenized = tokenizer(prompt_text, add_special_tokens=False)
    return len(tokenized["input_ids"])


def _text_token_ids(tokenizer, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    return [int(token_id) for token_id in tokenized["input_ids"]]


def _find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _valid_token_bounds(attention_mask: torch.Tensor | None, row_idx: int, seq_width: int) -> tuple[int, int]:
    if attention_mask is None:
        return 0, seq_width

    valid_positions = torch.nonzero(attention_mask[row_idx], as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return 0, 0
    return int(valid_positions[0].item()), int(valid_positions[-1].item()) + 1


def mask_labels_to_response(
    batch_inputs,
    tokenizer,
    prompt_texts,
    ignore_index: int = IGNORE_INDEX,
    response_texts: list[str] | None = None,
):
    """Keep loss only on assistant response tokens.

    ``prompt_texts`` must be produced from the same chat template as
    ``batch_inputs``, but without the assistant content and with
    ``add_generation_prompt=True``.

    When ``response_texts`` is provided, the response span is found by exact
    token matching inside the final multimodal ``input_ids``. This is required
    for Qwen-VL processors because image placeholders can expand to many
    ``<|image_pad|>`` tokens after chat-template tokenization, making plain
    prompt-text token counts too short.
    """
    input_ids = batch_inputs["input_ids"]
    labels = input_ids.clone()
    seq_width = labels.size(1)
    attention_mask = batch_inputs.get("attention_mask", None)

    if response_texts is not None:
        if len(response_texts) != labels.size(0):
            raise ValueError(f"Expected {labels.size(0)} response_texts, got {len(response_texts)}")

        labels.fill_(ignore_index)
        for i, response_text in enumerate(response_texts):
            valid_start, valid_end = _valid_token_bounds(attention_mask, i, seq_width)
            valid_ids = [int(token_id) for token_id in input_ids[i, valid_start:valid_end].tolist()]
            response_ids = _text_token_ids(tokenizer, str(response_text))
            relative_start = _find_last_subsequence(valid_ids, response_ids)
            if relative_start < 0:
                raise ValueError(
                    "Could not locate assistant response tokens in Qwen-VL input_ids. "
                    "This usually means the chat template transformed the assistant text unexpectedly."
                )

            response_start = valid_start + relative_start
            labels[i, response_start:valid_end] = input_ids[i, response_start:valid_end]

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = ignore_index
        return labels

    for i, prompt_text in enumerate(prompt_texts):
        prefix_len = _prompt_token_length(tokenizer, prompt_text)
        if attention_mask is not None:
            valid_start, _ = _valid_token_bounds(attention_mask, i, seq_width)
            response_start = valid_start + prefix_len
        else:
            response_start = prefix_len

        response_start = max(0, min(response_start, seq_width))
        labels[i, :response_start] = ignore_index

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = ignore_index

    return labels
