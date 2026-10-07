# Annotation guidelines

## Annotation method

This analysis compares Gemma and Qwen in non-thinking mode on
40 English examples from the 1,000-example dev set.

The sample contains:
- 20 representative examples
- 7 Gemma-better examples
- 7 Qwen-better examples
- 6 both-low examples

The selection groups reflect BERTScore-based sampling, not the
manual outcome labels. The combined sample was deliberately selected,
so its error frequencies should not be treated as estimates for the
whole dev set.

Annotations were entered manually using `annotate_error_analysis.py`.
The script displays examples, saves choices, and counts annotations;
it does not assign labels. ChatGPT was used to discuss the rubric and
individual annotation decisions.

Not all image-dependent claims were verified against the images.
Some cases may need further review. Reported counts reflect the
current saved annotations.

## Outcome rubric

1. **Correct**
   The answer correctly addresses all components of the question.
   It does not need to reproduce reference details that the question
   does not require.

2. **Partial**
   The central answer is correct, but a meaningful requested component
   is missing, or there is a significant qualification or error.

3. **Incorrect**
   The central answer is wrong, or the model fails to answer the
   actual question.

4. **Unverifiable**
   The answer makes a specific claim that the reference neither
   confirms nor contradicts, and available evidence, including image
   access, is insufficient to verify it.

The reference is the gold standard for this analysis, but it is not
necessarily an exhaustive description of the image. A detail missing
from the reference is not automatically false. An unverified extra
detail does not automatically make an otherwise correct answer
Unverifiable; consider its importance to the requested answer.

## Error taxonomy

### Assignment rules

- Assign one primary error type to each Partial or Incorrect answer.
- Correct and Unverifiable answers have `error_type = none`.
- Judge answers against the question and reference.
- The reference is not necessarily an exhaustive description of the image.
- Absence of a detail from the reference does not establish that it is false.
- Extra specificity is not automatically an error.

### Error types

1. `visual_identification`
   Wrong identification of an object, entity, building, landmark, flag, etc.

2. `ocr_text_interpretation`
   Misreading, mistranslating, or misinterpreting visible text.

3. `functional_semantic_interpretation`
   Wrong interpretation of purpose, function, event, significance, or meaning.

4. `cultural_contextual_knowledge`
   Wrong cultural, historical, religious, or regional interpretation.

5. `unsupported_inference_hallucination`
   A specific claim established as unsupported, invented, or contradicted.
   Do not assign this merely because the reference omits the claim.
   If the claim cannot be verified or contradicted with available evidence,
   apply the Unverifiable outcome where appropriate.

6. `omission_incomplete_answer`
   A meaningful component requested by the question is missing.

### Important distinctions

- Identifying a real but different entity usually falls under
  `visual_identification`.
- Inventing an entity or specific name falls under
  `unsupported_inference_hallucination` when fabrication is established.
- Naming or describing something without answering its requested purpose
  or significance usually falls under `omission_incomplete_answer`.
