# Baseline evaluation

Task: MMCultureQA Task 2 — Textual Visual QA.
Input: image and written question.
Output: short, open-ended textual answer.
Language: English.

Four baseline runs were completed on the same 1,000 dev examples.
Each model was evaluated in thinking and non-thinking modes.

## Results

The reported metric is mean BERTScore F1.

| Model | Mode | Examples | BERTScore F1 |
| --- | --- | ---: | ---: |
| Gemma | Non-thinking | 1,000 | 0.925133 |
| Gemma | Thinking | 1,000 | 0.924990 |
| Qwen | Non-thinking | 1,000 | 0.914081 |
| Qwen | Thinking | 1,000 | 0.900242 |

## Observations

Gemma's thinking and non-thinking scores are very close.
Thinking mode changes its score by -0.000143.

Qwen's thinking score is lower than its non-thinking score
by 0.013839.

These aggregate scores do not explain the causes of model errors
and should not be interpreted as percentages of correct answers.

## Qualitative analysis

The current manual error analysis compares Gemma and Qwen
in non-thinking mode on a selected 40-example sample.

- [Annotation guidelines](error_taxonomy.md)
- [40-example analysis results](error_analysis_40_results.md)
