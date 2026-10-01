import json
from pathlib import Path

from bert_score import score

FILES = {
    "gemma_nonthinking": "results/gemma_nonthinking_dev.jsonl",
    "gemma_thinking": "results/gemma_thinking_dev.jsonl",
    "qwen_nonthinking": "results/qwen_nonthinking_dev.jsonl",
    "qwen_thinking": "results/qwen_thinking_dev.jsonl",
}

results = {}

for name, file in FILES.items():
    candidates = []
    references = []

    with open(file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            candidates.append(item["prediction"])
            references.append(item["reference"])

    print(f"\n===== {name} =====")
    print(f"Examples: {len(candidates)}")

    _, _, F1 = score(
        candidates,
        references,
        lang="en",
        device="cuda",
        verbose=True,
        idf=False,
        rescale_with_baseline=False,
    )

    bertscore_f1 = F1.mean().item()

    results[name] = {
        "n": len(candidates),
        "bertscore_f1": bertscore_f1,
    }

    print(f"BERTScore F1: {bertscore_f1:.6f}")

output = Path("results/bertscore_baselines.json")

output.write_text(
    json.dumps(
        {
            "metric": "BERTScore F1",
            "lang": "en",
            "idf": False,
            "rescale_with_baseline": False,
            "runs": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nSaved results to {output}")
