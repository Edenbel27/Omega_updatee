# Frame Relation Classification

This document describes the frame-relation classifier implemented in OmegaClaw.
It explains the change from the original LLM-only classifier, the current
execution flow, the major files and functions, the symbolic reasoning layer,
and the next steps needed to cover more relation scenarios.

## Overview

The classifier compares a current frame with relevant historical frames and
returns `Relation` atoms containing:

- `FrameID-1`
- `FrameID-2`
- relation `Class`
- a human-readable `Reason`
- a numeric `Confidence`

The current implementation combines four kinds of evidence:

1. semantic vector retrieval from ChromaDB;
2. vector threshold detection for likely duplicates;
3. structural and content-based symbolic reasoning through PeTTa and NAL;
4. an LLM fallback for cases that remain ambiguous.

## Previous Structure: LLM-Only Classification

Previously, the flow was:

```text
current frame
    -> create or query embeddings
    -> retrieve nearby historical frames
    -> send every candidate to the chat LLM
    -> parse the LLM JSON response
    -> convert the response into Relation atoms
```

The implementation was centered on `_call_classifier_llm()` in
`src/frame_relation.py`. The LLM received frame fields, prose, candidate
metadata, and retrieved documents, then selected a relation class.

This had several drawbacks:

- every candidate batch required a chat-model call;
- obvious facts such as an explicit parent link were not resolved locally;
- equivalent inputs could receive different results;
- the reasoning was difficult to audit because the main decision was a JSON
  response from the LLM;
- token use and latency were incurred even for deterministic cases.

## Current Structure: Tiered Classification

The current flow is:

```text
frame parsing
    -> semantic document construction
    -> embedding and ChromaDB retrieval
    -> Tier 2: vector DuplicateOf detection
    -> Tier 1: structural MeTTa rules
    -> deep lib_nal content reasoning
    -> Tier 3: LLM fallback
    -> Relation S-expression output
```

A candidate resolved by an earlier tier is not sent to the later tier for the
same category. The external interface remains compatible with the existing
MeTTa caller.

The call originates in `src/context.metta`, which calls
`cfv2_compose_frame_relations()` in `src/frame_relation.py`.

## Main Files

| File                              | Responsibility                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `src/frame_relation.py`           | Parses frames, builds semantic documents, manages embeddings and ChromaDB, orchestrates the tiers, validates LLM results, and serializes relations.                                         |
| `src/frame_nal.py`                | Python-to-PeTTa adapter. Loads the external PeTTa runtime and MeTTa files, grounds Python frames as facts, invokes structural rules and real `lib_nal.metta` inference, and parses results. |
| `frame_nal.metta`                 | Frame-specific MeTTa rules for structural relation classification.                                                                                                                          |
| `lib_nal.metta`                   | Shared NAL truth functions and inference rules, including deduction, induction, abduction, analogy, revision, and the `                                                                     | -` operator. |
| `tests/test_frame_nal_adapter.py` | Focused tests for PeTTa loading, structural rules, content reasoning, dependency reasoning, and Relation parsing.                                                                           |

## Semantic Embeddings

### Semantic document

`_frame_document()` in `src/frame_relation.py` creates the text sent to the
embedding provider:

```text
Task: Fix authentication timeout.
Results: Login still fails after deployment.
Dependencies: DatabaseMigration.
```

The semantic document focuses on `deliverable`, `results`, and `dependencies`.
The frame ID, parent ID, status, source, mode, and priority are kept outside the
main semantic text because they are structural evidence rather than the meaning
of the task.

This separates two questions:

```text
embedding search: Which historical frames concern similar work?
symbolic reasoning: What relation follows from the available evidence?
```

### Embedding providers

For the OpenAI provider, `_embed_texts_openai()` calls the configured embedding
model, whose default is `text-embedding-3-large`:

```python
client.embeddings.create(
    model=FRAME_EMBED_MODEL,
    input=texts,
)
```

For the local provider, `_embed_texts_local()` calls the project embedding
bridge:

```python
lib_llm_ext.useLocalEmbedding(text)
```

The embedding model converts the semantic document into a numeric vector.
ChromaDB stores the vector and returns nearby historical vectors when a new
frame is queried. The vector identifies candidates; it does not directly decide
whether the relation is `FollowUp`, `Blocks`, or `Supersedes`.

### Vector duplicate tier

`_classify_relations_vector()` checks retrieved Chroma distances before running
symbolic or LLM classification. The configurable thresholds are:

```text
FRAME_DUPLICATE_DISTANCE_OPENAI=0.15
FRAME_DUPLICATE_DISTANCE_LOCAL=0.10
```

A candidate below the selected threshold is classified as `DuplicateOf`, with a
confidence derived from its distance, and is removed from the unresolved set.

The current conversion from distance to semantic score is:

```python
1.0 - distance / 2.0
```

This is a bounded heuristic, not an NAL formula. It assumes a distance range
approximately from `0` to `2`; the Chroma metric should be explicitly confirmed
and calibrated before treating the score as a scientific probability.

## Chroma Metadata and Content Hashing

The Chroma document contains semantic content. Metadata preserves information
needed after retrieval:

```text
frameID
parentID
status
priority
source
mode
dependencies
embeddingProvider
contentHash
```

`_frame_metadata()` stores these values so a retrieved candidate can be rebuilt
for symbolic reasoning.

The embedding input is hashed by `_hash_text()` using SHA-256. The hash includes:

```text
embedding provider + embedding model + semantic document
```

`_upsert_changed_frames()` compares the current hash with the stored
`contentHash`. If unchanged, it skips recomputing the vector. If the task,
results, dependencies, provider, or model changes, the hash changes and the
frame is embedded and upserted again.

This is an embedding cache and invalidation mechanism. It reduces repeated
embedding work and API cost; it does not classify relations.

## Structural Symbolic Reasoning

`src/frame_nal.py` converts known fields into MeTTa facts. Examples include:

```metta
((--> FrameA (status Active)) (stv 1.0 0.9))
((--> FrameA (parent FrameB)) (stv 1.0 0.9))
((--> FrameA (depends FrameB)) (stv 1.0 0.9))
```

It also creates pair-level evidence when appropriate:

```metta
((same-source FrameA FrameB) (stv 1.0 0.9))
((same-mode FrameA FrameB) (stv 1.0 0.9))
((top-level FrameA) (stv 1.0 0.9))
```

`build_query()` sends these facts to:

```metta
!(classify-frame-pair FrameA FrameB Score Facts)
```

`frame_nal.metta` currently contains these structural rules:

- `SubgoalOf`: the query frame explicitly has the candidate as its parent;
- `ParentOf`: the inverse parent relationship;
- `FollowUp`: the query has a completed candidate as its parent;
- `ContinuationOf`: a top-level frame shares source and mode with completed work;
- `SameProject`: top-level active frames share a source;
- `DependsOn`: the query lists the candidate as a dependency;
- `Blocks`: the query depends on an active candidate.

These rules are deterministic symbolic rules. Their input truth values are
currently assigned by the adapter because the fields are explicitly present in
the frame representation.

## Deep NAL Content Reasoning

NAL cannot reason directly over arbitrary English prose. The adapter first
creates symbolic content evidence.

### Concept grounding

`_content_concepts()` combines:

- `deliverable`;
- `results`;
- `dependencies`.

It lowercases the text, extracts normalized words, and removes a small set of
stopwords.

Example:

```text
current: implement authentication timeout fix
history: investigate authentication timeout
```

Shared concepts may include:

```text
authentication
timeout
```

The adapter creates NAL premises:

```metta
((--> Current authentication) (stv 1.0 0.75))
((--> History authentication) (stv 1.0 0.75))
```

### Actual `lib_nal.metta` inference

For every grounded shared concept, `infer_content_relations()` invokes the
actual NAL operator loaded from `lib_nal.metta`:

```metta
!(|-
  ((--> Current authentication) (stv 1.0 0.75))
  ((--> History authentication) (stv 1.0 0.75)))
```

The PeTTa runtime executes the NAL rules and returns derived truth values. The
adapter parses those values and aggregates support across concepts. Multiple
concepts add a limited support bonus, while the final confidence is capped.

If the embedding score is high but literal words do not overlap, the adapter
adds a `semantic-neighbor` concept. This allows semantically close frames with
different wording to enter the NAL content stage.

The current content stage produces `RelatedButSeparate`. It is genuine NAL
inference over grounded facts, but the grounding is intentionally lightweight:
it uses shared normalized concepts and embedding-neighbor evidence rather than a
full natural-language proposition extractor.

## Python-to-PeTTa Runtime

`src/frame_nal.py` searches for the external PeTTa installation through
`PETTA_PATH`, then uses the configured fallback path if available. It creates a
persistent PeTTa runtime and loads:

```text
lib_nal.metta
frame_nal.metta
```

The runtime chain is:

```text
frame_relation.py
    -> frame_nal.py
    -> petta.py
    -> janus_swi
    -> SWI-Prolog
    -> MeTTa rules and lib_nal.metta
```

A persistent runtime avoids starting Prolog and loading the MeTTa files for
every candidate pair.

## LLM Fallback

The existing LLM classifier is retained in `_classify_relations_llm()`. It is
called only for candidates unresolved by vector and NAL stages.

The LLM remains useful for relations that require deeper prose interpretation,
including:

- `Supersedes`, where intent and contradiction must be understood;
- `SameFailureCluster`, where root-cause analysis is required;
- nuanced semantic relations not represented in the current ontology;
- incomplete or contradictory evidence.

The LLM is now a final fallback rather than the default classifier.

## Missing Fields

Frames without a `frameID` are skipped because no relation can reference them.
Missing `parentID`, `status`, `source`, `mode`, and `dependencies` do not crash
the parser.

When evidence is missing:

```text
missing parentID       -> parent facts are unavailable; frame may be top-level
missing status         -> status-dependent rules cannot fire
missing source/mode    -> same-source or same-mode rules cannot fire
missing dependencies   -> dependency rules cannot fire
missing prose          -> content grounding may produce no concepts
```

Unresolved candidates continue to the later tiers. Incomplete frames should be
handled conservatively because the absence of a field is not proof that the
opposite relationship is false.

## Tests

The focused adapter suite is:

```powershell
pytest tests/test_frame_nal_adapter.py -q
```

It currently covers:

- parent and follow-up rules;
- semantic relations with different structural fields;
- content NAL reasoning without parent fields;
- embedding-supported reasoning with different wording;
- dependency and blocking rules;
- Relation parsing.

The current focused result is:

```text
6 passed
```

## Next Symbolic Reasoning Work

The current implementation establishes the Python, PeTTa, MeTTa, and NAL
integration. More scenarios require richer grounding and more explicit NAL
relations.

### 1. Replace token overlap with an ontology

Simple words are not enough for robust reasoning. Introduce controlled concepts
for:

```text
actions, goals, entities, failures, causes, states, outputs
```

For example, map `login`, `sign-in`, and `authentication` into a controlled
concept family instead of relying on literal word overlap.

### 2. Ground relation propositions

The adapter currently grounds concepts, not propositions such as:

```text
continues(FrameA, FrameB)
depends-on(FrameA, FrameB)
supersedes(FrameA, FrameB)
caused-by(FrameA, FrameB)
contradicts(FrameA, FrameB)
```

These propositions are required for content-derived `FollowUp`, `Supersedes`,
`SameFailureCluster`, and `Blocks`.

### 3. Add relation-specific NAL rules

Add MeTTa rules and NAL truth calculations for:

- content-derived follow-up without `parentID`;
- `Supersedes` based on action and contradiction evidence;
- `SameFailureCluster` based on shared failure causes;
- content-derived `DependsOn` and `Blocks`;
- revision when multiple frames provide conflicting evidence.

### 4. Use multi-hop inference

Once propositions are grounded, use NAL deduction, induction, abduction,
analogy, and revision across multiple hops. Keep hop limits and confidence
thresholds explicit because truth confidence decreases as inference chains grow.

### 5. Define relation precedence

A pair may satisfy multiple rules. Define whether the system should return all
valid categories or select one strongest category. A possible precedence is:

```text
DuplicateOf > FollowUp > SubgoalOf > ParentOf
Blocks > DependsOn
ContinuationOf > SameProject > RelatedButSeparate
```

### 6. Improve missing-field safety

Do not encode missing `source`, `status`, or `mode` as if `UNKNOWN` were an
actual shared value. Track field presence explicitly and avoid duplicate-vector
classification when both frames have no meaningful semantic content.

### 7. Add end-to-end tests

Add tests through `cfv2_compose_frame_relations()` using a mocked or live Chroma
collection. Verify:

- the vector tier resolves duplicates;
- NAL resolves structural and content relations;
- unresolved candidates reach the LLM;
- missing fields do not create false relations;
- final Relation S-expressions remain compatible with `context.metta`.
