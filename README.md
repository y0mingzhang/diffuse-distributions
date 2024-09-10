# Forcing Diffuse Distributions out of Language Models

## What is this?

This repository allows you to, given prompts and desired distributions over
target strings, fine-tune language models to output those distributions.

Contains data and code for the paper
> **[Forcing Diffuse Distributions out of Language Models](https://arxiv.org/abs/2404.10859)**  
> Yiming Zhang, Avi Schwarzschild, Nicholas Carlini, Zico Kolter and Daphne Ippolito  
> COLM 2024

## Installing

Python 3.11 (tested) + `pip install -r requirements.txt`.

## Running the code

### Prompts

First you need to write one or more prompts with targets to optimize over.
See `prompts/numbers/rng-0.json` for a prompt describing a random number
generator of numbers 1-10.

Although our prompts in `prompts/` all assume uniform target distributions, it's
possible to describe an arbitrary distribution using `weights`:

```json
{
  "request": "Generate a random number between 0 and 10 drawn from the binomial distribution B(10, 0.2). Output only the number between two curly braces, like this: {number}. Don't output code.",
  "response": [
    ""
  ],
  "targets": [
    "{0}",
    "{1}",
    "{2}",
    "{3}",
    "{4}",
    "{5}",
    "{6}",
    "{7}",
    "{8}",
    "{9}",
    "{10}"
  ],
  "weights": [
    0.107374182,
    0.268435456,
    0.301989888,
    0.201326592,
    0.088080384,
    0.0264241152,
    0.005505024,
    0.000786432,
    7.3728e-05,
    4.096e-06,
    1.024e-07
  ]
}
```

### Configuration

We use a YAML config file to configure a fine-tuning experiment:

```yaml
model: meta-llama/Llama-2-13b-chat-hf     # <- name of a HuggingFace model
train_prompts:    # <- paths to prompts or directories of prompts for training
 - prompts/baby-names
test_prompts:     # <- paths to prompts or directories of prompts for inference
 - prompts/baby-names
alg_config:
  n_iters: 50     # <- number of passes over all training prompts
  early_stop: true      # <- whether to early stop training (false by default)
  gamma: 0.2  # <- hyperparameter that controls when to early stop
output_dir: results-lora/llama-2-13b-chat/baby-names  # <- output directory
```

Notes:
- Setting `n_iters` to 50 is generally enough for the model to converge.
- Set `early_stop` to `true` when you do not have an exhaustive
target set (e.g., when optimizing the model to produce diverse baby names).
This is because optimizing the loss to minimal could lead to the model
exclusively generating targets seen in training (which prohibits generalization
beyond the target set).
When `early_stop` is `true`, training stops when `loss <= optimal_loss * (1 + gamma)`.

### Training

Run `python src/train.py $CONFIG`, where `$CONFIG` is a YAML config.
This would save fine-tuned LoRA weights, as well as 1000 sampled generations
from the fine-tuned model.

### Generation

**Sampling from the base model**: `python src/generate.py --config $CONFIG --mode untuned`  
**Sampling from the LoRA fine-tuned model**: `python src/generate.py --config $CONFIG --mode lora`

See `src/generate.py` for additional arguments.

## License

MIT

## Citation

```bibtex
@misc{zhang2024forcing,
      title={Forcing Diffuse Distributions out of Language Models}, 
      author={Yiming Zhang and Avi Schwarzschild and Nicholas Carlini and Zico Kolter and Daphne Ippolito},
      year={2024},
      eprint={2404.10859},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
