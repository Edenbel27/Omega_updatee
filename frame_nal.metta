;; Facts are passed as a list by src/frame_nal.py. Semantic similarity is
;; passed separately because PeTTa represents generated lists as Prolog lists.

(= (fact-present $Facts $Wanted $Result)
(let $Fact (superpose $Facts)
(if (== $Fact $Wanted)
    $Result
(empty)
)
)
)

(= (and4 $A $B $C $D)
(and $A (and $B (and $C $D))))

(= (and5 $A $B $C $D $E)
(and $A (and $B (and $C (and $D $E)))))

(= (content-relation $Facts $F1 $F2 $Predicate $Class $Reason $FactConfidence $OutputConfidence)
    (fact-present $Facts
    (content-proposition $F1 $F2 $Predicate)
        (Relation
        (FrameID-1 $F1)
        (FrameID-2 $F2)
        (Class $Class)
        (Reason $Reason)
        (Confidence $OutputConfidence))))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
        (classify-semantic $F1 $F2 $Score))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
        (classify-structural $F1 $F2 $Facts))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
            (content-relation $Facts $F1 $F2 continues FollowUp
            "content indicates that the current frame continues prior work" 0.82 0.82))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
            (content-relation $Facts $F1 $F2 follows-work FollowUp
            "content indicates follow-up work without a parent field" 0.78 0.78))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
            (content-relation $Facts $F1 $F2 supersedes Supersedes
            "content indicates that the current frame replaces prior work" 0.84 0.84))

        (= (classify-frame-pair $F1 $F2 $Score $Facts)
            (content-relation $Facts $F1 $F2 same-failure SameFailureCluster
            "content indicates a shared failure cluster" 0.80 0.80))

            ;; Strong semantic similarity means the frames concern the same work.
            ;; The conclusion remains RelatedButSeparate unless a stronger structural
            ;; conclusion is available.
        (= (classify-semantic $F1 $F2 $Score)
        (if (> $Score 0.65)
            (Relation
            (FrameID-1 $F1)
            (FrameID-2 $F2)
            (Class RelatedButSeparate)
            (Reason "semantic similarity inferred from prior frame")
            (Confidence $Score))
            (empty)
            )
        )

        ;; Parent relation.
    (= (classify-structural $F1 $F2 $Facts)
        (fact-present $Facts
        ((--> $F1 (parent $F2)) (stv 1.0 0.9))
            (Relation
            (FrameID-1 $F1)
            (FrameID-2 $F2)
            (Class SubgoalOf)
            (Reason "NAL inferred an explicit parent relation")
            (Confidence 1.0))
            )
        )

        ;; A completed parent frame represents a follow-up rather than only a subgoal.
    (= (classify-structural $F1 $F2 $Facts)
        (if (and
        (fact-present $Facts ((--> $F1 (parent $F2)) (stv 1.0 0.9)) True)
        (fact-present $Facts ((--> $F2 (status Completed)) (stv 1.0 0.9)) True))
            (Relation
            (FrameID-1 $F1)
            (FrameID-2 $F2)
            (Class FollowUp)
            (Reason "NAL inferred a follow-up from parent and completed status")
            (Confidence 1.0))
            (empty)))

                ;; Reverse parent relation.
            (= (classify-structural $F1 $F2 $Facts)
                (fact-present $Facts
                ((--> $F2 (parent $F1)) (stv 1.0 0.9))
                    (Relation
                    (FrameID-1 $F1)
                    (FrameID-2 $F2)
                    (Class ParentOf)
                    (Reason "NAL inferred the inverse parent relation")
                    (Confidence 1.0)))
                    )

                    ;; A top-level frame with the same source and mode can continue completed work.
                (= (classify-structural $F1 $F2 $Facts)
                    (if (and4
                    (fact-present $Facts ((top-level $F1) (stv 1.0 0.9)) True)
                    (fact-present $Facts ((same-source $F1 $F2) (stv 1.0 0.9)) True)
                    (fact-present $Facts ((same-mode $F1 $F2) (stv 1.0 0.9)) True)
                    (fact-present $Facts ((--> $F2 (status Completed)) (stv 1.0 0.9)) True))
                        (Relation
                        (FrameID-1 $F1)
                        (FrameID-2 $F2)
                        (Class ContinuationOf)
                        (Reason "NAL inferred continuation from source, mode, and completed status")
                        (Confidence 0.85))
                        (empty)))

                            ;; Independent active frames from the same project are related but separate.
                        (= (classify-structural $F1 $F2 $Facts)
                            (if (and5
                            (fact-present $Facts ((top-level $F1) (stv 1.0 0.9)) True)
                            (fact-present $Facts ((top-level $F2) (stv 1.0 0.9)) True)
                            (fact-present $Facts ((same-source $F1 $F2) (stv 1.0 0.9)) True)
                            (fact-present $Facts ((--> $F1 (status Active)) (stv 1.0 0.9)) True)
                            (fact-present $Facts ((--> $F2 (status Active)) (stv 1.0 0.9)) True))
                                (Relation
                                (FrameID-1 $F1)
                                (FrameID-2 $F2)
                                (Class SameProject)
                                (Reason "NAL inferred shared source and independent active work")
                                (Confidence 0.75))
                                (empty)
                                )
                            )

                            ;; Dependency evidence can identify a relation even when parent/source/mode
                            ;; rules do not match.
                        (= (classify-structural $F1 $F2 $Facts)
                            (fact-present $Facts
                            ((--> $F1 (depends $F2)) (stv 1.0 0.9))
                                (Relation
                                (FrameID-1 $F1)
                                (FrameID-2 $F2)
                                (Class DependsOn)
                                (Reason "NAL inferred a dependency from the frame evidence")
                                (Confidence 1.0))))

                                    ;; A dependency on an active frame blocks the dependent frame.
                                (= (classify-structural $F1 $F2 $Facts)
                                    (if (and
                                    (fact-present $Facts ((--> $F1 (depends $F2)) (stv 1.0 0.9)) True)
                                    (fact-present $Facts ((--> $F2 (status Active)) (stv 1.0 0.9)) True))
                                        (Relation
                                        (FrameID-1 $F1)
                                        (FrameID-2 $F2)
                                        (Class Blocks)
                                        (Reason "NAL inferred that an active dependency blocks progress")
                                        (Confidence 1.0))
                                        (empty)))