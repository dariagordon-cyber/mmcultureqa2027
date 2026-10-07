# Error analysis: 40 examples

Manual annotations comparing Gemma and Qwen in non-thinking mode.
Outcome counts and error frequencies describe this sample only.
The sample includes representative examples and cases selected
for model differences or low BERTScore; it is exploratory.

Annotation rules: [Error taxonomy](error_taxonomy.md).

## Sample composition

| Selection group | Examples |
| --- | ---: |
| both_low | 6 |
| gemma_better | 7 |
| qwen_better | 7 |
| representative | 20 |

## Outcomes

| Outcome | Gemma | Qwen |
| --- | ---: | ---: |
| correct | 24 | 19 |
| partial | 5 | 9 |
| incorrect | 8 | 8 |
| unverifiable | 3 | 4 |

## Primary error types

One primary error per Partial or Incorrect answer.
Correct and Unverifiable answers have error_type = none.

| Error type | Gemma | Qwen |
| --- | ---: | ---: |
| visual_identification | 5 | 7 |
| ocr_text_interpretation | 1 | 2 |
| functional_semantic_interpretation | 0 | 0 |
| cultural_contextual_knowledge | 0 | 2 |
| unsupported_inference_hallucination | 0 | 2 |
| omission_incomplete_answer | 7 | 4 |

These are the current saved labels. Some uncertain cases may
require review with the images before drawing stronger conclusions.
