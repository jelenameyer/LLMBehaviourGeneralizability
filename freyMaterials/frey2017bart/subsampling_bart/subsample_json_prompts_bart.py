import random

input_file = "prompts.jsonl"
output_file = "subsample_prompts_bart.jsonl"
n = 4

# Count lines first
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

sampled_lines = random.sample(lines, n)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(sampled_lines)