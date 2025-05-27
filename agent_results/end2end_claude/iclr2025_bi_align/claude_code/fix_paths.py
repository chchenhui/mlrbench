#!/usr/bin/env python3
import re

# Path to the results.md file
results_md_path = '/home/chenhui/mlr-bench/pipeline_claude/iclr2025_bi_align/results/results.md'

# Read the current content
with open(results_md_path, 'r') as f:
    content = f.read()

# Replace all image paths
corrected_content = re.sub(
    r'!\[(.*?)\]\(\.\.\/(.*?)\.png\)',
    r'![\1](./figures/\2.png)',
    content
)

# Write the corrected content back
with open(results_md_path, 'w') as f:
    f.write(corrected_content)

print("All image paths have been fixed in results.md")