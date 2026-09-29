from __future__ import annotations

import os
import pathlib
import re
import sys
from typing import Any


_HERE = pathlib.Path(__file__).resolve()
_OMEGA_ROOT = _HERE.parents[1]
_PETTA_CANDIDATES = (
    os.environ.get("PETTA_PATH", ""),
    r"C:\Users\lenovo\OneDrive\Desktop\icog\Training\Tr1\disjoint_set_PeTTa_1",
)
_PETTA_PATH = next(
    (
        pathlib.Path(value).expanduser()
        for value in _PETTA_CANDIDATES
        if value and pathlib.Path(value).expanduser().is_dir()
    ),
    None,
)

if _PETTA_PATH:
    petta_python = _PETTA_PATH / "python"
    if str(petta_python) not in sys.path:
        sys.path.append(str(petta_python))

try:
    import petta
except ImportError as exc:
    petta = None
    _PETTA_IMPORT_ERROR = exc
else:
    _PETTA_IMPORT_ERROR = None


_METTA = None
_NAL_LOADED = False


def _runtime():
    global _METTA, _NAL_LOADED

    if petta is None:
        raise RuntimeError(
            "PeTTa is unavailable. Install/configure PETTA_PATH before "
            "using NAL frame classification."
        ) from _PETTA_IMPORT_ERROR

    if _METTA is None:
        _METTA = petta.PeTTa(
            verbose=False,
            petta_path=_PETTA_PATH,
        )

    if not _NAL_LOADED:
        # PeTTa embeds paths in Prolog quoted atoms; use POSIX separators on Windows.
        nal_path = (_OMEGA_ROOT / "lib_nal.metta").as_posix()
        frame_nal_path = (_OMEGA_ROOT / "frame_nal.metta").as_posix()
        _METTA.load_metta_file(nal_path)
        _METTA.load_metta_file(frame_nal_path)
        _NAL_LOADED = True

    return _METTA


def infer_relations(
    query_frame: dict[str, Any],
    candidate_frames: list[dict[str, Any]],
    relation_classes: list[str],
) -> list[dict[str, Any]]:
    relations = []
    for candidate_frame in candidate_frames:
        expression = build_query(query_frame, candidate_frame)
        result = _runtime().process_metta_string(expression)
        relations.extend(
            relation for relation in parse_relations(result)
            if relation["class"] in relation_classes
        )
        relations.extend(
            relation
            for relation in infer_content_relations(
                query_frame,
                candidate_frame,
                relation_classes,
            )
            if relation["class"] in relation_classes
        )
    return relations


def infer_content_relations(
    query_frame: dict[str, Any],
    candidate_frame: dict[str, Any],
    relation_classes: list[str],
) -> list[dict[str, Any]]:
    """Use real lib_nal |- abduction over concepts grounded from frame prose."""
    query_id = _symbol(query_frame["frameID"])
    candidate_id = _symbol(candidate_frame["frameID"])
    query_concepts = _content_concepts(query_frame)
    candidate_concepts = _content_concepts(candidate_frame)
    shared_concepts = sorted(query_concepts & candidate_concepts)
    semantic_score = max(
        0.0,
        min(1.0, float(candidate_frame.get("semanticScore", 0.0))),
    )
    if semantic_score >= 0.65:
        shared_concepts.append("semantic-neighbor")
    if not shared_concepts:
        return []

    runtime = _runtime()
    conclusions = []
    for concept in shared_concepts:
        premise_a = _nal_inheritance(query_id, concept, 0.75)
        premise_b = _nal_inheritance(candidate_id, concept, 0.75)
        expression = f"!(|- {premise_a} {premise_b})"
        result = runtime.process_metta_string(expression)
        conclusions.extend(_parse_nal_conclusions(result))

    if not conclusions or "RelatedButSeparate" not in relation_classes:
        return []

    confidence = _aggregate_nal_confidence(conclusions, len(shared_concepts))
    return [{
        "frameID1": query_id,
        "frameID2": candidate_id,
        "class": "RelatedButSeparate",
        "reason": (
            "lib_nal abduction derived a relation from shared content concepts: "
            + ", ".join(shared_concepts[:8])
        ),
        "confidence": confidence,
    }]


def _content_concepts(frame: dict[str, Any]) -> set[str]:
    text = " ".join((
        str(frame.get("deliverable", "")),
        str(frame.get("results", "")),
        str(frame.get("dependencies", "")),
    )).lower()
    words = re.findall(r"[a-z][a-z0-9_]{2,}", text)
    stopwords = {
        "and", "are", "for", "from", "into", "that", "the", "this",
        "with", "will", "have", "has", "was", "were", "then", "when",
        "after", "before", "completed", "active", "task", "result",
    }
    return {word for word in words if word not in stopwords}


def _nal_inheritance(frame_id: str, concept: str, confidence: float) -> str:
    return f"((--> {frame_id} {concept}) (stv 1.0 {confidence:.4f}))"


def _parse_nal_conclusions(result: Any) -> list[tuple[float, float]]:
    text = str(result)
    values = re.findall(
        r"\(stv\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\)",
        text,
    )
    return [(float(frequency), float(confidence)) for frequency, confidence in values]


def _aggregate_nal_confidence(
    conclusions: list[tuple[float, float]],
    shared_count: int,
) -> float:
    best = max(confidence for _, confidence in conclusions)
    support_bonus = min(0.15, max(0, shared_count - 1) * 0.05)
    return min(0.95, max(0.0, best + support_bonus))


def build_query(
    query_frame: dict[str, Any],
    candidate_frame: dict[str, Any],
) -> str:
    query_id = _symbol(query_frame["frameID"])
    candidate_id = _symbol(candidate_frame["frameID"])

    query_facts = _frame_facts(query_frame)
    candidate_facts = _frame_facts(candidate_frame)

    semantic_score = max(
        0.0,
        min(1.0, float(candidate_frame.get("semanticScore", 0.0))),
    )
    semantic_fact = (
        f"((--> {query_id} (semantic {candidate_id})) "
        f"(stv {semantic_score:.4f} 0.85))"
    )

    facts = " ".join(query_facts + candidate_facts + [semantic_fact])

    pair_facts = []
    if query_frame.get("source") and query_frame.get("source") == candidate_frame.get("source"):
        pair_facts.append(
            f"((same-source {query_id} {candidate_id}) (stv 1.0 0.9))"
        )
    if query_frame.get("mode") and query_frame.get("mode") == candidate_frame.get("mode"):
        pair_facts.append(
            f"((same-mode {query_id} {candidate_id}) (stv 1.0 0.9))"
        )
    if _symbol(query_frame.get("parentID", "")) == "UNKNOWN":
        pair_facts.append(
            f"((top-level {query_id}) (stv 1.0 0.9))"
        )
    if _symbol(candidate_frame.get("parentID", "")) == "UNKNOWN":
        pair_facts.append(
            f"((top-level {candidate_id}) (stv 1.0 0.9))"
        )
    facts = " ".join([facts, *pair_facts])

    return (
        f"!(classify-frame-pair {query_id} {candidate_id} "
        f"{semantic_score:.4f} ({facts}))"
    )


def _frame_facts(frame: dict[str, Any]) -> list[str]:
    frame_id = _symbol(frame["frameID"])
    facts = []

    for field in ("source", "status", "mode"):
        value = _symbol(frame.get(field, "Unknown"))
        facts.append(
            f"((--> {frame_id} ({field} {value})) "
            f"(stv 1.0 0.9))"
        )

    parent_id = _symbol(frame.get("parentID", "UNKNOWN"))
    if parent_id != "UNKNOWN":
        facts.append(
            f"((--> {frame_id} (parent {parent_id})) "
            f"(stv 1.0 0.9))"
        )

    dependencies = str(frame.get("dependencies", ""))
    for dependency in dependencies.split():
        dependency_id = _symbol(dependency)
        if dependency_id != "UNKNOWN":
            facts.append(
                f"((--> {frame_id} (depends {dependency_id})) "
                f"(stv 1.0 0.9))"
            )

    return facts


def _symbol(value: Any) -> str:
    text = str(value)
    return "".join(
        char for char in text
        if char.isalnum() or char in "_-:"
    ) or "UNKNOWN"


def parse_relations(result: Any) -> list[dict[str, Any]]:
    # Replace this with a structured PeTTa result parser once the
    # runtime result shape is confirmed in the local environment.
    text = str(result)
    return parse_relation_text(text)


def parse_relation_text(text: str) -> list[dict[str, Any]]:
    relations = []

    # Temporary parser for the stable Relation output format.
    # Prefer a PeTTa atom parser when available.
    import re

    pattern = re.compile(
        r"\(Relation\s+"
        r"\(FrameID-1\s+([A-Za-z0-9_:-]+)\)\s+"
        r"\(FrameID-2\s+([A-Za-z0-9_:-]+)\)\s+"
        r"\(Class\s+([A-Za-z0-9_-]+)\)\s+"
        r"\(Reason\s+\"((?:\\.|[^\"])*)\"\)\s+"
        r"\(Confidence\s+([0-9.]+)\)\)"
    )

    for match in pattern.finditer(text):
        relations.append({
            "frameID1": match.group(1),
            "frameID2": match.group(2),
            "class": match.group(3),
            "reason": bytes(match.group(4), "utf-8").decode(
                "unicode_escape"
            ),
            "confidence": float(match.group(5)),
        })

    return _deduplicate_relations(relations)


def _deduplicate_relations(
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (
            str(relation.get("frameID1", "")),
            str(relation.get("frameID2", "")),
            str(relation.get("class", "")),
        )
        previous = unique.get(key)
        if previous is None or float(relation.get("confidence", 0.0)) > float(previous.get("confidence", 0.0)):
            unique[key] = relation
    return list(unique.values())