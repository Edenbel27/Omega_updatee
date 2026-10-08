from __future__ import annotations

import pathlib
import sys

import pytest


_OMEGA_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(_OMEGA_ROOT / "providers"))

frame_nal = pytest.importorskip("frame_nal")
frame_relation = pytest.importorskip("frame_relation")


def _frame(frame_id, *, source="User", status="Active", mode="Fast", parent="", dependencies=""):
    return {
        "frameID": frame_id,
        "source": source,
        "status": status,
        "mode": mode,
        "parentID": parent,
        "dependencies": dependencies,
        "semanticScore": 0.1,
    }


def test_parent_and_follow_up_rules():
    result = frame_nal.infer_relations(
        _frame("Child", parent="Parent"),
        [_frame("Parent", status="Completed")],
        ["SubgoalOf", "FollowUp"],
    )

    classes = {relation["class"] for relation in result}
    assert {"SubgoalOf", "FollowUp"} <= classes


def test_semantic_relation_works_when_fields_differ():
    result = frame_nal.infer_relations(
        _frame("New", source="User", mode="Fast"),
        [_frame("Old", source="System", status="Completed", mode="Deep") | {"semanticScore": 0.82}],
        ["RelatedButSeparate"],
    )

    assert result[0]["class"] == "RelatedButSeparate"
    assert result[0]["confidence"] == pytest.approx(0.82)


def test_deep_nal_uses_content_without_parent_fields():
    result = frame_nal.infer_content_relations(
        _frame("Current", source="User", mode="Fast") | {
            "deliverable": "implement authentication timeout fix",
            "results": "login failure after deployment",
        },
        _frame("History", source="System", status="Completed", mode="Deep") | {
            "deliverable": "investigate authentication timeout",
            "results": "login failure after deployment",
            "semanticScore": 0.2,
        },
        ["RelatedButSeparate"],
    )

    assert result[0]["class"] == "RelatedButSeparate"
    assert "lib_nal abduction" in result[0]["reason"]


def test_deep_nal_uses_embedding_evidence_for_different_words():
    result = frame_nal.infer_content_relations(
        _frame("Current") | {
            "deliverable": "repair sign-in timeout",
            "results": "requests fail after release",
            "semanticScore": 0.82,
        },
        _frame("History", status="Completed") | {
            "deliverable": "resolve authentication latency",
            "results": "service degraded after deployment",
            "semanticScore": 0.82,
        },
        ["RelatedButSeparate"],
    )

    assert result[0]["class"] == "RelatedButSeparate"
    assert "semantic-neighbor" in result[0]["reason"]


def test_dependency_and_blocking_rules():
    result = frame_nal.infer_relations(
        _frame("Blocked", dependencies="Dependency"),
        [_frame("Dependency", status="Active")],
        ["DependsOn", "Blocks"],
    )

    classes = {relation["class"] for relation in result}
    assert {"DependsOn", "Blocks"} <= classes


def test_content_propositions_infer_parentless_follow_up():
    result = frame_nal.infer_relations(
        _frame("Current") | {
            "deliverable": "continue implementing authentication fix",
        },
        [_frame("History", status="Completed") | {
            "deliverable": "investigate authentication failure",
        }],
        ["FollowUp"],
    )

    assert result[0]["class"] == "FollowUp"
    assert "continues prior work" in result[0]["reason"]


def test_content_propositions_infer_supersedes():
    result = frame_nal.infer_relations(
        _frame("Current") | {
            "deliverable": "revert deployment change",
        },
        [_frame("History", status="Completed") | {
            "deliverable": "implement deployment change",
        }],
        ["Supersedes"],
    )

    assert result[0]["class"] == "Supersedes"


def test_content_propositions_infer_same_failure_cluster():
    result = frame_nal.infer_relations(
        _frame("Current") | {
            "deliverable": "fix authentication failure",
        },
        [_frame("History") | {
            "deliverable": "investigate authentication failure",
        }],
        ["SameFailureCluster"],
    )

    assert result[0]["class"] == "SameFailureCluster"


def test_relation_precedence_selects_strongest_category():
    from frame_relation import _select_preferred_relations

    result = _select_preferred_relations([
        {
            "frameID1": "Current",
            "frameID2": "History",
            "class": "SubgoalOf",
            "confidence": 1.0,
        },
        {
            "frameID1": "Current",
            "frameID2": "History",
            "class": "FollowUp",
            "confidence": 0.8,
        },
    ])

    assert [relation["class"] for relation in result] == ["FollowUp"]


def test_relation_text_parser_preserves_reason():
    text = (
        '(Relation (FrameID-1 New) (FrameID-2 Old) '
        '(Class FollowUp) (Reason "completed parent") (Confidence 1.0))'
    )

    assert frame_nal.parse_relation_text(text) == [{
        "frameID1": "New",
        "frameID2": "Old",
        "class": "FollowUp",
        "reason": "completed parent",
        "confidence": 1.0,
    }]


def test_vector_tier_passes_duplicate_threshold():
    resolved, unresolved = frame_relation._classify_relations_vector(
        [{"frameID": "Old", "distance": 0.05}],
        ["DuplicateOf"],
        "OpenAI",
    )

    assert resolved[0]["class"] == "DuplicateOf"
    assert unresolved == []


def test_vector_tier_rejects_non_duplicate_distance():
    resolved, unresolved = frame_relation._classify_relations_vector(
        [{"frameID": "Old", "distance": 0.40}],
        ["DuplicateOf"],
        "OpenAI",
    )

    assert resolved == []
    assert unresolved[0]["frameID"] == "Old"


def test_structural_nal_fails_without_parent_evidence():
    result = frame_nal.infer_relations(
        _frame("Current"),
        [_frame("History")],
        ["SubgoalOf", "ParentOf"],
    )

    assert result == []


def test_content_nal_fails_without_shared_evidence():
    result = frame_nal.infer_content_relations(
        _frame("Current") | {"deliverable": "write weather report", "semanticScore": 0.1},
        _frame("History") | {"deliverable": "repair database migration", "semanticScore": 0.1},
        ["RelatedButSeparate"],
    )

    assert result == []


def test_unresolved_candidate_reaches_llm(monkeypatch):
    hits = [{"frameID": "History", "distance": 0.4}]
    calls = []

    monkeypatch.setattr(
        frame_relation,
        "_classify_relations_vector",
        lambda *args: ([], hits),
    )
    monkeypatch.setattr(
        frame_relation,
        "_classify_relations_nal",
        lambda *args: [],
    )

    def fake_llm(*args):
        calls.append(args)
        return [{
            "frameID1": "Current",
            "frameID2": "History",
            "class": "Supersedes",
            "reason": "LLM fallback",
            "confidence": 0.7,
        }]

    monkeypatch.setattr(frame_relation, "_classify_relations_llm", fake_llm)
    result = frame_relation._classify_relations(
        {"frameID": "Current"},
        hits,
        ["Supersedes"],
        "OpenAI",
    )

    assert calls
    assert result[0]["class"] == "Supersedes"


def test_resolved_nal_candidate_skips_llm(monkeypatch):
    hits = [{"frameID": "History", "distance": 0.4}]
    monkeypatch.setattr(
        frame_relation,
        "_classify_relations_vector",
        lambda *args: ([], hits),
    )
    monkeypatch.setattr(
        frame_relation,
        "_classify_relations_nal",
        lambda *args: [{
            "frameID1": "Current",
            "frameID2": "History",
            "class": "FollowUp",
            "reason": "NAL",
            "confidence": 0.9,
        }],
    )
    monkeypatch.setattr(
        frame_relation,
        "_classify_relations_llm",
        lambda *args: pytest.fail("resolved NAL candidate reached LLM"),
    )

    result = frame_relation._classify_relations(
        {"frameID": "Current"},
        hits,
        ["FollowUp"],
        "OpenAI",
    )

    assert result[0]["class"] == "FollowUp"


def test_parent_of_rule_passes_for_inverse_parent_fact():
    result = frame_nal.infer_relations(
        _frame("Current"),
        [_frame("Parent", parent="Current")],
        ["ParentOf"],
    )

    assert result[0]["class"] == "ParentOf"


def test_continuation_of_rule_passes_for_matching_completed_top_level_frame():
    result = frame_nal.infer_relations(
        _frame("Current", source="User", mode="Fast"),
        [_frame("History", source="User", status="Completed", mode="Fast")],
        ["ContinuationOf"],
    )

    assert result[0]["class"] == "ContinuationOf"


def test_same_project_rule_passes_for_independent_active_frames():
    result = frame_nal.infer_relations(
        _frame("Current", source="User", status="Active"),
        [_frame("History", source="User", status="Active")],
        ["SameProject"],
    )

    assert result[0]["class"] == "SameProject"


def test_content_hash_reuses_unchanged_embedding(monkeypatch):
    frame = _frame("Current") | {
        "deliverable": "fix authentication",
        "results": "failure remains",
        "dependencies": "Database",
    }
    document = frame_relation._frame_document(frame)
    content_hash = frame_relation._hash_text(
        f"OpenAI:{frame_relation.FRAME_EMBED_MODEL}:{document}"
    )

    class Collection:
        def get(self, **kwargs):
            return {"ids": ["Current"], "metadatas": [{"contentHash": content_hash}]}

        def upsert(self, **kwargs):
            pytest.fail("unchanged frame was re-embedded")

    monkeypatch.setattr(frame_relation, "_get_collection", lambda provider: Collection())
    monkeypatch.setattr(
        frame_relation,
        "_embed_texts",
        lambda *args: pytest.fail("unchanged frame requested embedding"),
    )

    assert frame_relation._upsert_changed_frames([frame], "OpenAI") == {}


def test_content_hash_reembeds_changed_document(monkeypatch):
    frame = _frame("Current") | {"deliverable": "new task"}
    calls = []

    class Collection:
        def get(self, **kwargs):
            return {"ids": ["Current"], "metadatas": [{"contentHash": "old-hash"}]}

        def upsert(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(frame_relation, "_get_collection", lambda provider: Collection())
    monkeypatch.setattr(frame_relation, "_embed_texts", lambda texts, provider: [[0.1, 0.2]])

    result = frame_relation._upsert_changed_frames([frame], "OpenAI")

    assert result["Current"] == [0.1, 0.2]
    assert len(calls) == 1
    assert calls[0]["ids"] == ["Current"]


def test_chroma_search_returns_nearest_historical_frame(monkeypatch):
    class Collection:
        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "ids": [["Current", "History"]],
                "documents": [["current document", "history document"]],
                "metadatas": [[{}, {"status": "Completed"}]],
                "distances": [[0.0, 0.12]],
            }

    monkeypatch.setattr(frame_relation, "_get_collection", lambda provider: Collection())
    hits = frame_relation._search_top_k(
        {"frameID": "Current"},
        [0.1, 0.2],
        "OpenAI",
        5,
    )

    assert [hit["frameID"] for hit in hits] == ["History"]
    assert hits[0]["distance"] == pytest.approx(0.12)


def test_missing_fields_do_not_create_parent_or_dependency_facts():
    facts = frame_nal._frame_facts({
        "frameID": "Current",
        "source": "",
        "status": "",
        "mode": "",
        "parentID": "",
        "dependencies": "",
    })

    assert not any("(parent " in fact for fact in facts)
    assert not any("(depends " in fact for fact in facts)


def test_compose_frame_relations_serializes_end_to_end_result(monkeypatch):
    compact_frames = """
    ((Frame (frameID Current) (parentID ()) (source User)
      (priority 1.0) (status Active) (frame-mode Fast)
      (deliverables (\"current task\")) (results ())))
    """
    calls = []
    monkeypatch.setattr(
        frame_relation,
        "_upsert_changed_frames",
        lambda frames, provider: {"Current": [0.1, 0.2]},
    )
    monkeypatch.setattr(
        frame_relation,
        "_search_top_k",
        lambda *args: [{
            "frameID": "History",
            "distance": 0.2,
            "semanticScore": 0.9,
            "document": "history task",
            "metadata": {},
        }],
    )

    def fake_classify(query, hits, classes, provider):
        calls.append((query["frameID"], hits[0]["frameID"], provider))
        return [{
            "frameID1": "Current",
            "frameID2": "History",
            "class": "RelatedButSeparate",
            "reason": "test result",
            "confidence": 0.8,
        }]

    monkeypatch.setattr(frame_relation, "_classify_relations", fake_classify)
    result = frame_relation.cfv2_compose_frame_relations(
        compact_frames,
        "Current",
        "(RelatedButSeparate)",
        "Local",
        5,
    )

    assert calls == [("Current", "History", "Local")]
    assert "(Class RelatedButSeparate)" in result
    assert "(Confidence 0.8000)" in result
