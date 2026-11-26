# Decision memo: runbook question answering

## Use case and data

Support engineers need answers grounded in eight local incident runbooks and the
25-question evaluation set in `data/docs_qa.csv`. The runbooks change as operating
procedures change, so the answer must reflect current material and cite its source.

## Recommendation

Use retrieval-augmented generation (RAG) with a small local vector index and
source/section metadata. Prompting alone cannot supply the private runbook facts.
A long context window is a reasonable baseline at this corpus size, but does not
provide robust evidence selection as the corpus grows. Fine-tuning is not
appropriate: the facts change and must remain attributable.

## Cost estimate

The local Ollama path has no per-token API charge; budget for the operator's
existing laptop and electricity. For a hosted model, estimate monthly spend as
`queries × (input_tokens × input_price + output_tokens × output_price) / 1,000,000`.
For 250 queries/month at 2,000 input and 250 output tokens, that is 0.5625M
billable tokens; use the provider's current published prices before enabling
cloud calls. No billing is required for the local exercise.

## Success measures

Ship only after the 25-question benchmark reports hit rate@4, citation correctness,
and average latency for each chunk setting. Require at least 80% retrieval hit@4
and 100% valid source references on the reviewed evaluation set. Re-run the
benchmark after every runbook or embedding-model change.
