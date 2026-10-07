# Error taxonomy

Used for manual analysis of Gemma and Qwen non-thinking answers.

## Assignment rules

- Assign one primary error type to each Partial or Incorrect answer.
- Correct and Unverifiable answers have `error_type = none`.
- Judge answers against the question and reference.
- The reference is not necessarily an exhaustive description of the image.
- Absence of a detail from the reference does not establish that it is false.
- Extra specificity is not automatically an error.

## Error types

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

## Important distinctions

- Identifying a real but different entity usually falls under
  `visual_identification`.
- Inventing an entity or specific name falls under
  `unsupported_inference_hallucination` when fabrication is established.
- Naming or describing something without answering its requested purpose
  or significance usually falls under `omission_incomplete_answer`.
