# Experiment Results

## Models Compared
- BART Large CNN (baseline)
- LED Base 16384 (proposed)

## Rouge Scores
| name           |   rouge1 |    rouge2 |    rougel |
|:---------------|---------:|----------:|----------:|
| bart-large-cnn | 0.313043 | 0.125093  | 0.214245  |
| led-base-16384 | 0.159307 | 0.0573432 | 0.0998238 |

![ROUGE Comparison](../figures/comparison_rouge.png)

## Time and Memory
| name           |   avg_time_sec |   max_memory_bytes |
|:---------------|---------------:|-------------------:|
| bart-large-cnn |       0.603081 |         1775901696 |
| led-base-16384 |       2.02923  |         2282892288 |

![Time and Memory](../figures/time_memory.png)
