import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import Prompt, embed_tokens, entropy


def compute_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[Prompt],
    batch_size: int = 32,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device="cuda")

    for prompt in prompts:
        prompt_embeds = embed_tokens(model)(torch.tensor(prompt.tokens, device="cuda"))

        target_weights = torch.tensor(
            prompt.weights, dtype=torch.float32, device="cuda"
        )
        target_tokens = [random.choice(target) for target in prompt.targets]

        for offset in range(0, len(target_tokens), batch_size):
            target_tokens_batch = target_tokens[offset : offset + batch_size]
            inputs_embeds = []
            target_ids = []
            target_masks = []
            max_len = max(map(len, target_tokens_batch))
            prompt_len = prompt_embeds.shape[0]

            for target in target_tokens_batch:
                pad_len = max_len - len(target)
                ids = torch.tensor(
                    target + [tokenizer.eos_token_id] * pad_len,
                    dtype=torch.long,
                    device="cuda",
                )
                target_ids.append(ids)

                embeds = torch.cat((prompt_embeds, embed_tokens(model)(ids)))
                inputs_embeds.append(embeds)

                target_mask = torch.tensor(
                    [1] * len(target) + [0] * pad_len,
                    dtype=torch.float32,
                    device="cuda",
                )
                target_masks.append(target_mask)

            target_ids = torch.stack(target_ids)
            inputs_embeds = torch.stack(inputs_embeds)
            target_masks = torch.stack(target_masks)

            outputs = model(inputs_embeds=inputs_embeds)
            logits = rearrange(
                outputs.logits[:, prompt_len - 1 : -1, :], "B S V -> (B S) V"
            )
            token_negative_log_likelihoods = F.cross_entropy(
                logits, rearrange(target_ids, "B S -> (B S)"), reduction="none"
            )
            sequence_negative_log_likelihoods = (
                rearrange(
                    token_negative_log_likelihoods,
                    "(B S) -> B S",
                    B=len(target_tokens_batch),
                )
                * target_masks
            ).sum(1)

            loss_batch = (
                sequence_negative_log_likelihoods
                * target_weights[offset : offset + batch_size]
            ).sum() / len(prompts)
            loss = loss + loss_batch

    return loss


def continuous(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[Prompt],
    n_iters: int = 10000,
    batch_size: int = 32,
    early_stop: bool = False,
    gamma: float = 0.2,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-5)

    for _ in (pbar := trange(n_iters)):
        random.shuffle(prompts)
        loss = 0.0
        optimal_loss = 0.0
        for prompt in prompts:
            num_targets = len(prompt.targets)
            rand_indices = list(range(num_targets))
            random.shuffle(rand_indices)

            optimal_loss += entropy(np.array(prompt.weights))
            for offset in range(0, num_targets, batch_size):
                prompt_fed = Prompt(
                    prompt.alias,
                    prompt.tokens,
                    [
                        prompt.targets[i]
                        for i in rand_indices[offset : offset + batch_size]
                    ],
                    [
                        prompt.weights[i]
                        for i in rand_indices[offset : offset + batch_size]
                    ],
                )
                loss_batch = compute_loss(
                    model,
                    tokenizer,
                    [prompt_fed],
                )
                loss_batch.backward()
                loss += loss_batch.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        pbar.set_postfix(
            {
                "loss": f"{loss:.2f}",
                "optimal": f"{optimal_loss:.2f}",
            }
        )

        if early_stop and loss <= optimal_loss * (1 + gamma):
            print(f"early stopping @ loss {loss:.2f}")
            break
    model.eval()
