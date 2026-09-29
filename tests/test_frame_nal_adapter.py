from __future__ import annotations

import pytest


frame_nal = pytest.importorskip("frame_nal")


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
