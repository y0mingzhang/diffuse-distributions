import argparse
import json
import os

import torch
from omegaconf import OmegaConf
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import Prompt, find_and_generate_prompts


PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@torch.inference_mode(True)
def generate(
    model: AutoModelForCausalLM | PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[Prompt],
    reps: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, list[str]]:
    generations: dict[str, list[str]] = {}
    with tqdm(total=batch_size * reps * len(prompts), desc="sampling..") as pbar:
        for prompt in prompts:
            generations[prompt.alias] = []

            generation_config = {
                "do_sample": True,
                "temperature": 1.0,
                "top_k": None,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            }

            tokens = torch.tensor(prompt.tokens).to("cuda")
            batch = {"input_ids": tokens.repeat(batch_size, 1)}
            seq_len = tokens.shape[0]
            for rep in range(reps):
                outputs = model.generate(**batch, **generation_config)
                generations[prompt.alias].extend(
                    tokenizer.batch_decode(
                        outputs[:, seq_len:], skip_special_tokens=True
                    )
                )
                pbar.update(batch_size)

    return generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="path to the yaml config", required=True
    )
    parser.add_argument(
        "--mode",
        choices=["lora", "untuned"],
        help="generating with LoRA, soft prompt or the original model?",
        default="untuned",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="where to dump the generations",
        default="generations.json",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        help="the number of generations for each prompt. should be divisible by batch size",
        default=1000,
    )
    parser.add_argument(
        "--n_batch_size", type=int, help="batch size of generation", default=20
    )
    parser.add_argument(
        "--max_new_tokens", type=int, help="max number of tokens to sample", default=50
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS.keys(),
        help="model tensor precision",
        default="bf16",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    pretrained_model_name = config.model

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name, use_fast=False, legacy=True
    )
    test_prompts = find_and_generate_prompts(config.test_prompts, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name, device_map="auto", torch_dtype=PRECISIONS[args.precision]
    )

    if args.mode == "lora":
        model = PeftModelForCausalLM.from_pretrained(
            model=model, model_id=config.output_dir
        )
    reps = args.n_generations // args.n_batch_size
    generations = generate(
        model,
        tokenizer,
        test_prompts,
        reps,
        args.n_batch_size,
        args.max_new_tokens,
    )

    output_path = args.output_file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Writing generations to {output_path}")
    with open(output_path, "w") as f:
        json.dump(generations, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
