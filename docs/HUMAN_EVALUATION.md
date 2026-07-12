# Blinded human evaluation

The human-study tool deliberately separates collection from scoring. Do not give
raters the answer key or the source video filenames.

```python
from src.utils.human_eval import HumanEvaluationFramework

study = HumanEvaluationFramework("artifacts/human-study")
study.create_evaluation_clips(real_videos, generated_videos, seed=42)
study.save_evaluation_protocol()
study.materialize_blinded_media()
study.build_web_ui()
```

Serve `artifacts/human-study/` with a local HTTP server and give raters only the
served directory. The public protocol refers to neutral paths such as
`media/clip_0000_left.mp4`; the private source pairing remains only in the
operator's process and `answer_key.json`.

The browser UI exports unscored `responses_<rater>.json`. After collection,
load each export privately and call `score_blinded_responses`; only then may
correctness and aggregate accuracy be computed. Store both the public protocol
and private answer key with the final report so the randomization is auditable.

This creates tooling, not evidence: no study result should be represented as a
GameNGen result until participant, clip, model, data, and protocol artifacts
are preserved.
