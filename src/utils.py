import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaTokenizer


@dataclass
class Prompt:
    alias: str
    tokens: list[int]
    targets: list[list[list[int]]]
    weights: list[float]

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def generate_prompt(
    tokenizer: AutoTokenizer,
    alias: str,
    request: str,
    response: str | list[str],
    targets: list[str],
    weights: list[float] | None = None,
    eos: bool = True,
) -> Prompt:
    if isinstance(response, str):
        response = [response]
    assert isinstance(response, list) and isinstance(response[0], str)

    tokens = tokenizer.apply_chat_template(
        [{"content": request, "role": "user"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    targets_encodings = []
    for target in targets:
        if "Mistral-7B-Instruct" in tokenizer.name_or_path:
            # if we don't do this, mistral tokenizer would tokenize [\INST]{1} into [\ | INST | ]{ | 1 | }
            target = " " + target

        target_encodings = []
        for resp in response:
            full_encoding = tokenizer.apply_chat_template(
                [
                    {"content": request, "role": "user"},
                    {"content": resp + target, "role": "assistant"},
                ],
                tokenize=True,
            )

            if eos:
                if isinstance(tokenizer, GemmaTokenizer):
                    # Gemma doesn't like <end_of_turn> and always produces <eos>
                    assert full_encoding[-2:] == [107, 108]
                    target_encodings.append(full_encoding[len(tokens) : -2] + [1])
                else:
                    target_encodings.append(full_encoding[len(tokens) :])
            else:
                target_encoding = tokenizer.encode(
                    resp + target, add_special_tokens=False
                )
                assert all(
                    a == b for a, b in zip(tokens + target_encoding, full_encoding)
                )
                target_encodings.append(target_encoding)
        targets_encodings.append(target_encodings)

    if not weights:
        weights = [1.0 / len(targets) for _ in range(len(targets))]
    else:
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]

    return Prompt(alias, tokens, targets_encodings, weights)


def entropy(
    p: np.ndarray,
) -> float:
    return -np.nansum(p * np.log(p))


def find_and_generate_prompts(
    paths: Iterable[str], tokenizer: AutoTokenizer | None
) -> list[Prompt | dict]:
    prompt_paths = []
    for path in paths:
        if os.path.isfile(path):
            prompt_paths.append(path)
        else:
            assert os.path.isdir(path), path
            for prompt_path in glob.glob(
                os.path.join(path, "**", "*.json"), recursive=True
            ):
                prompt_paths.append(prompt_path)

    prompts = []
    alias_set = set()
    for prompt_path in prompt_paths:
        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)
        alias = prompt_path
        assert alias not in alias_set
        alias_set.add(alias)
        if tokenizer is None:
            prompts.append({"alias": alias} | prompt_data)
        else:
            prompts.append(generate_prompt(tokenizer, alias, **prompt_data))
    return prompts


def embed_tokens(model: AutoModelForCausalLM | PeftModelForCausalLM):
    if isinstance(model, PeftModelForCausalLM):
        return embed_tokens(model.base_model.model)
    return model.model.embed_tokens
