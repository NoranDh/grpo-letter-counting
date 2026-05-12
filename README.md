# GRPO Fine-Tuning for Letter Counting

Fine-tuning a large language model to accurately count letter occurrences in words using Group Relative Policy Optimization (GRPO).

## Overview

This project applies GRPO — a reinforcement learning-based fine-tuning technique — to teach **Qwen2.5-3B-Instruct** to count how many times a given letter appears in a word. The model learns through four custom reward functions rather than supervised labels, eliminating the need for a second evaluator model.

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a fine-tuning approach where the model generates multiple responses to the same prompt, and a set of reward functions scores each response. The model is then updated to produce higher-scoring responses over time — similar to how a student improves by getting feedback on multiple attempts at the same problem.

## Reward Functions

Four reward functions were implemented to guide the model's learning:

| Reward Function | What it evaluates |
|---|---|
| `format_reward_func` | Whether the response uses `<reasoning>` and `<answer>` tags correctly, and whether the answer is an integer |
| `numbering_reward_func` | Whether the model numbers each letter in order (1, 2, 3...) without skipping or repeating |
| `spelling_reward_func` | Whether the model spells the word correctly letter by letter |
| `counting_reward_func` | Whether the running count at each step is accurate |

## Training

- **Base model:** Qwen2.5-3B-Instruct (4-bit quantized)
- **Method:** GRPO with LoRA (rank 8)
- **Hardware:** NVIDIA T4 GPU
- **Steps:** 100
- **Key parameters:** `learning_rate=5e-6`, `beta=0.04`, `num_generations=4`

## Results

The reward curve below shows the model improving over 100 training steps. The orange line (correct answer reward mean) trends upward from ~0.2 to ~1.0, indicating the model learned to count letters more accurately.

![Training Reward Curve](reward_curve.png)

## Stack

- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA fine-tuning
- [TRL](https://github.com/huggingface/trl) — GRPO trainer
- [vLLM](https://github.com/vllm-project/vllm) — fast inference during training
- HuggingFace Datasets & Transformers
