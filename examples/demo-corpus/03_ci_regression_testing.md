# CI Regression Testing for RAG Pipelines

Continuous Integration (CI) regression testing for RAG systems ensures that changes to the retriever, embedding model, generator, or prompt templates do not silently degrade quality. RAGEvalKit provides a `rageval ci-check` command that compares a candidate run against a baseline run and fails with exit code 1 if configured thresholds are violated.

## Why CI Testing Matters for RAG

RAG pipelines are composed of multiple independently-changeable components. A change to the chunking strategy, embedding model, or prompt template can improve quality on some questions while degrading it on others. Without systematic regression testing, regressions go undetected until they cause visible product issues. CI gates create a safety net by encoding minimum quality standards and maximum allowed regression tolerances.

## Absolute Thresholds

Absolute thresholds define a minimum acceptable score for each metric, regardless of what the baseline achieves. They are configured under the `absolute` key in a thresholds YAML file:

```yaml
absolute:
  recall_at_k_min: 0.70
  answer_relevance_min: 0.70
  faithfulness_min: 0.80
```

If the candidate run achieves a mean faithfulness of 0.75 and the configured minimum is 0.80, the CI check fails. Absolute thresholds are useful for enforcing a quality floor — the system must always meet a baseline level of quality, even if no regression occurred.

## Relative Thresholds

Relative thresholds define the maximum allowed drop from baseline to candidate. They are configured under the `relative` key:

```yaml
relative:
  recall_at_k_drop_max: 0.05
  answer_relevance_drop_max: 0.05
  faithfulness_drop_max: 0.05
  mrr_drop_max: 0.05
```

A drop is computed as `baseline_mean - candidate_mean`. If the baseline achieves faithfulness of 0.90 and the candidate achieves 0.82, the drop is 0.08, which exceeds the allowed 0.05 and triggers a failure. Relative thresholds catch regressions even when the absolute scores are still above the floor.

## Backward-Compatible Aliases

For compatibility with older configurations, `retrieval_relevance_min` is treated as an alias for `recall_at_k_min`, and `retrieval_relevance_drop_max` is treated as an alias for `recall_at_k_drop_max`. When both are present, the primary name (`recall_at_k_min` or `recall_at_k_drop_max`) takes precedence.

## Exit Codes and CI Integration

The `rageval ci-check` command returns:
- **Exit code 0**: all configured thresholds are satisfied. The candidate run passes the CI gate.
- **Exit code 1**: one or more thresholds are violated. The CI gate fails.

This makes it straightforward to integrate with CI systems such as GitHub Actions, GitLab CI, or Jenkins. Any non-zero exit code will mark the CI step as failed, preventing a regression from being merged.

Use `--json` to emit a machine-readable JSON result alongside the human-readable table:

```bash
rageval ci-check --baseline $BASELINE_RUN_ID --candidate $CANDIDATE_RUN_ID \
  --thresholds rageval.yaml --json
```

The JSON output includes `passed`, `violations` (a list of threshold violations with metric name, check type, threshold, and actual value), and the run IDs.

## Comparing Runs

Before using `ci-check`, use `rageval compare` to get a human-readable diff of the two runs. `compare` shows metric deltas, overall label count changes, and root-cause distribution differences. This is useful for understanding the full picture before deciding whether a regression is acceptable.

## Recommended CI Workflow

1. Run the full evaluation pipeline on the baseline commit and store the run ID.
2. Make changes to the RAG pipeline (embedding model, prompt, retrieval parameters).
3. Run the full evaluation pipeline on the candidate commit.
4. Run `rageval ci-check` with the stored baseline run ID and the new candidate run ID.
5. If the check fails, examine violations and use `rageval compare` and `rageval report` to understand the root cause.
6. Either fix the regression or, if the trade-off is intentional, update the thresholds YAML and the baseline run ID.
