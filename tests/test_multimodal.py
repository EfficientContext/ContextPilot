"""Tests for contextpilot.multimodal (image / video-frame context blocks)."""

import base64
import copy

import pytest

from contextpilot.multimodal import (
    ImageBlock,
    build_multimodal_messages,
    build_order_hint,
    extract_image_blocks,
    frame_label,
    optimize_multimodal,
    optimize_multimodal_batch,
    reconstruct_image_blocks,
    reorder_blocks,
    reorder_blocks_batch,
    true_order,
)
from contextpilot.multimodal.intercept import image_key, order_from_label
from contextpilot.server.intercept_parser import parse_intercept_headers
from contextpilot.server.live_index import ContextPilot


def _blk(i: int, ts=None, label=True) -> ImageBlock:
    """Tiny fake frame i with content-derived key."""
    data = f"frame-{i}".encode()
    return ImageBlock.from_bytes(
        data,
        order=float(ts if ts is not None else i),
        label=frame_label(i, ts) if label else "",
    )


def _images(msgs):
    """Return list of image_url values in the user message, in order."""
    user = [m for m in msgs if m["role"] == "user"][-1]
    return [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]


def _texts(msgs):
    user = [m for m in msgs if m["role"] == "user"][-1]
    return [p["text"] for p in user["content"] if p["type"] == "text"]


# ── blocks ───────────────────────────────────────────────────────────────


class TestImageBlock:
    def test_from_bytes_key_is_content_hash(self):
        a = ImageBlock.from_bytes(b"xyz")
        b = ImageBlock.from_bytes(b"xyz", order=3.0, label="different label")
        c = ImageBlock.from_bytes(b"xyzw")
        assert a.key == b.key and a == b
        assert a.key != c.key
        assert a.image_url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(a.image_url.split(",", 1)[1]) == b"xyz"

    def test_from_path_and_label(self, tmp_path):
        p = tmp_path / "f.png"
        p.write_bytes(b"\x89PNG fake")
        blk = ImageBlock.from_path(p, order=1.5, label=frame_label(0, 1.5))
        assert blk.image_url.startswith("data:image/png;base64,")
        assert blk.meta["path"] == str(p)
        assert blk.label == "[Frame 1 | t=1.5s]"

    def test_to_content_parts(self):
        blk = ImageBlock.from_bytes(b"a", label="L", detail="low")
        parts = blk.to_content_parts()
        assert parts[0] == {"type": "text", "text": "L"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["detail"] == "low"
        assert blk.to_content_parts(include_label=False)[0]["type"] == "image_url"

    def test_video_frames_to_blocks(self, tmp_path):
        from contextpilot.multimodal import video_frames_to_blocks

        paths = []
        for i in range(3):
            p = tmp_path / f"{i}.jpg"
            p.write_bytes(f"img{i}".encode())
            paths.append(p)
        blocks = video_frames_to_blocks(paths, [0.0, 2.0, 4.0], video_id="v1")
        assert [b.key for b in blocks] == ["v1#0", "v1#1", "v1#2"]
        assert [b.order for b in blocks] == [0.0, 2.0, 4.0]
        assert blocks[1].label == "[Frame 2 | t=2.0s]"


# ── prompt ───────────────────────────────────────────────────────────────


class TestPrompt:
    def test_true_order_and_hint(self):
        b0, b1, b2 = _blk(0), _blk(1), _blk(2)
        displayed = [b2, b0, b1]
        assert [b.order for b in true_order(displayed)] == [0.0, 1.0, 2.0]
        hint = build_order_hint(displayed)
        assert "NOT shown in chronological order" in hint
        assert hint.endswith("[Frame 1], [Frame 2], [Frame 3].")

    def test_hint_without_labels_uses_display_positions(self):
        displayed = [_blk(2, label=False), _blk(0, label=False), _blk(1, label=False)]
        hint = build_order_hint(displayed)
        assert hint.endswith("image 2, image 3, image 1.")

    def test_hint_in_order(self):
        hint = build_order_hint([_blk(0), _blk(1)])
        assert "are shown in chronological order" in hint
        assert build_order_hint([_blk(0), _blk(1)], in_order_template=None) == ""

    def test_hint_no_order_info(self):
        blocks = [ImageBlock.from_bytes(b"a"), ImageBlock.from_bytes(b"b")]
        assert build_order_hint(blocks) == ""

    @pytest.mark.parametrize("mode", ["sentence", "labels", "both", "none"])
    def test_modes(self, mode):
        displayed = [_blk(1), _blk(0)]
        msgs = build_multimodal_messages(displayed, "Q?", order_hint=mode)
        assert msgs[0]["role"] == "system"
        texts = _texts(msgs)
        # labels live in the parts before the trailing text (hint + question)
        has_labels = any(t in ("[Frame 1]", "[Frame 2]") for t in texts[:-1])
        has_sentence = "true chronological order" in texts[-1]
        assert has_labels == (mode in ("labels", "both"))
        assert has_sentence == (mode in ("sentence", "both"))
        # question is always the last text part
        assert texts[-1].endswith("Q?")
        # hint (if any) precedes the question inside the same trailing text part
        if has_sentence:
            assert texts[-1].index("chronological") < texts[-1].index("Q?")

    def test_prefix_is_position_independent(self):
        """Same displayed frames -> identical parts before the trailing text."""
        b = [_blk(i) for i in range(4)]
        m1 = build_multimodal_messages(b, "first question?", order_hint="both")
        m2 = build_multimodal_messages(b, "second question?", order_hint="both")
        c1, c2 = m1[-1]["content"], m2[-1]["content"]
        assert c1[:-1] == c2[:-1]
        assert c1[-1] != c2[-1]

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            build_multimodal_messages([_blk(0), _blk(1)], "q", order_hint="bogus")

    def test_intro_and_extra_suffix(self):
        msgs = build_multimodal_messages(
            [_blk(0), _blk(1)], "Q", intro="INTRO", extra_suffix="Answer with a letter.",
            system_instruction=None,
        )
        assert msgs[0]["role"] == "user"
        texts = _texts(msgs)
        assert texts[0] == "INTRO"
        assert texts[-1].endswith("Q\n\nAnswer with a letter.")


# ── API (real ContextPilot index) ────────────────────────────────────────


class TestReorderAPI:
    def test_single_request_first_call_unchanged(self):
        pilot = ContextPilot(use_gpu=False)
        blocks = [_blk(3), _blk(1), _blk(2)]
        out = reorder_blocks(blocks, pilot=pilot)
        assert [b.key for b in out] == [b.key for b in blocks]

    def test_dedup_within_request(self):
        pilot = ContextPilot(use_gpu=False)
        b = _blk(0)
        out = reorder_blocks([b, _blk(1), b], pilot=pilot)
        assert len(out) == 2

    def test_second_request_shares_prefix(self):
        """Frames shared with a previous request move to the front."""
        pilot = ContextPilot(use_gpu=False)
        f = [_blk(i) for i in range(6)]
        first = reorder_blocks([f[0], f[1], f[2], f[3]], pilot=pilot)
        # New request: shares {1,2,3} with the first but leads with new frames
        second = reorder_blocks([f[4], f[5], f[3], f[2], f[1]], pilot=pilot)
        second_keys = [b.key for b in second]
        shared = {f[1].key, f[2].key, f[3].key}
        assert set(second_keys) == {b.key for b in [f[1], f[2], f[3], f[4], f[5]]}
        # shared frames must form a contiguous prefix in the same relative
        # order as in the first request's (cached) prompt
        first_keys = [b.key for b in first]
        shared_in_first = [k for k in first_keys if k in shared]
        assert second_keys[: len(shared)] == shared_in_first

    def test_batch_reorder_and_schedule(self):
        pilot = ContextPilot(use_gpu=False)
        f = [_blk(i) for i in range(10)]
        reqs = [
            [f[0], f[1], f[2], f[3]],
            [f[7], f[8], f[9]],
            [f[3], f[2], f[1], f[4]],
            [f[9], f[7], f[8]],
        ]
        out, order = reorder_blocks_batch(reqs, pilot=pilot)
        assert sorted(order) == [0, 1, 2, 3]
        assert len(out) == 4
        for displayed, oi in zip(out, order):
            assert {b.key for b in displayed} == {b.key for b in reqs[oi]}
        # requests 0 & 2 share {1,2,3}: the shared frames must be a common prefix
        by_orig = {oi: [b.key for b in d] for d, oi in zip(out, order)}
        shared = {f[1].key, f[2].key, f[3].key}
        assert by_orig[0][:3] == by_orig[2][:3]
        assert set(by_orig[0][:3]) == shared

    def test_optimize_multimodal_baseline_is_chronological(self):
        pilot = ContextPilot(use_gpu=False)
        blocks = [_blk(2), _blk(0), _blk(1)]
        msgs = optimize_multimodal(blocks, "Q?", pilot=pilot, reorder=False)
        assert _images(msgs) == [b.image_url for b in true_order(blocks)]
        assert "are shown in chronological order" in _texts(msgs)[-1]

    def test_optimize_multimodal_reordered_with_sentence(self):
        pilot = ContextPilot(use_gpu=False)
        f = [_blk(i) for i in range(5)]
        # First request keeps the given (retrieval) order: [f2, f0]
        m1 = optimize_multimodal([f[2], f[0]], "Q1?", pilot=pilot)
        assert _images(m1) == [f[2].image_url, f[0].image_url]
        assert _texts(m1)[-1].startswith("Note: the frames above are NOT")
        assert "[Frame 1], [Frame 3]" in _texts(m1)[-1]
        # Second request shares {f0, f2}: they must lead, in the cached order
        msgs = optimize_multimodal([f[3], f[0], f[2]], "Q2?", pilot=pilot)
        imgs = _images(msgs)
        assert imgs[:2] == [f[2].image_url, f[0].image_url]
        tail = _texts(msgs)[-1]
        assert "true chronological order" in tail
        assert "[Frame 1], [Frame 3], [Frame 4]" in tail
        assert tail.endswith("Q2?")

    def test_optimize_multimodal_batch(self):
        pilot = ContextPilot(use_gpu=False)
        f = [_blk(i) for i in range(6)]
        reqs = [[f[0], f[1], f[2]], [f[2], f[1], f[5]]]
        batch, order = optimize_multimodal_batch(reqs, ["A?", "B?"], pilot=pilot)
        assert len(batch) == 2
        for msgs, oi in zip(batch, order):
            assert _texts(msgs)[-1].endswith(["A?", "B?"][oi])
            assert set(_images(msgs)) == {b.image_url for b in reqs[oi]}

    def test_batch_length_mismatch(self):
        with pytest.raises(ValueError):
            optimize_multimodal_batch([[_blk(0), _blk(1)]], ["a", "b"])


# ── intercept ────────────────────────────────────────────────────────────


def _mm_body(n=4, labels=True, question="Which happens first?"):
    content = [{"type": "text", "text": "Frames from a video:"}]
    for i in range(n):
        if labels:
            content.append({"type": "text", "text": f"[Frame {i+1} | t={i*2.0:.1f}s]"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{'A'*8}{i}"}}
        )
    content.append({"type": "text", "text": question})
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": content},
        ],
    }


class TestIntercept:
    def test_order_from_label(self):
        assert order_from_label("[Frame 3 | t=12.5s]") == 12.5
        assert order_from_label("Frame 7") == 7.0
        assert order_from_label("frame #2") == 2.0
        assert order_from_label("no info") is None
        assert order_from_label("") is None

    def test_extract_with_labels(self):
        ext = extract_image_blocks(_mm_body(4))
        assert ext is not None and ext.msg_index == 1
        assert len(ext.blocks) == 4
        assert ext.order_from_labels
        assert [b.order for b in ext.blocks] == [0.0, 2.0, 4.0, 6.0]
        assert ext.prefix_parts[0]["text"] == "Frames from a video:"
        assert ext.suffix_parts[0]["text"] == "Which happens first?"
        assert ext.blocks[2].label == "[Frame 3 | t=4.0s]"
        assert ext.blocks[0].key == image_key(ext.blocks[0].image_url)

    def test_extract_without_labels_uses_display_order(self):
        ext = extract_image_blocks(_mm_body(3, labels=False))
        assert ext is not None and not ext.order_from_labels
        assert [b.order for b in ext.blocks] == [0.0, 1.0, 2.0]

    def test_extract_none_when_single_image_or_text_only(self):
        assert extract_image_blocks(_mm_body(1)) is None
        assert extract_image_blocks({"messages": [{"role": "user", "content": "hi"}]}) is None
        assert extract_image_blocks({"messages": "nope"}) is None

    def test_reconstruct_roundtrip_identity(self):
        body = _mm_body(3)
        ext = extract_image_blocks(body)
        out = reconstruct_image_blocks(body, ext, ext.blocks, order_hint="none")
        assert out == body  # nothing changed, original untouched
        assert out is not body

    def test_reconstruct_reordered_with_hint(self):
        body = _mm_body(3)
        ext = extract_image_blocks(body)
        displayed = [ext.blocks[2], ext.blocks[0], ext.blocks[1]]
        out = reconstruct_image_blocks(body, ext, displayed)
        content = out["messages"][1]["content"]
        imgs = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
        assert imgs == [b.image_url for b in displayed]
        # labels travel with their image
        idx = [i for i, p in enumerate(content) if p["type"] == "image_url"]
        assert content[idx[0] - 1]["text"] == "[Frame 3 | t=4.0s]"
        # hint prepended to the question text part
        assert content[-1]["type"] == "text"
        assert content[-1]["text"].startswith("Note: the frames above are NOT")
        assert "[Frame 1 | t=0.0s], [Frame 2 | t=2.0s], [Frame 3 | t=4.0s]" in content[-1]["text"]
        assert content[-1]["text"].endswith("Which happens first?")
        assert content[0]["text"] == "Frames from a video:"
        # original body untouched
        assert body["messages"][1]["content"][-1]["text"] == "Which happens first?"

    def test_reconstruct_hint_when_no_suffix(self):
        body = _mm_body(2, question="")
        body["messages"][1]["content"].pop()  # drop question part entirely
        ext = extract_image_blocks(body)
        out = reconstruct_image_blocks(body, ext, [ext.blocks[1], ext.blocks[0]])
        assert out["messages"][1]["content"][-1]["type"] == "text"
        assert "true chronological order" in out["messages"][1]["content"][-1]["text"]

    def test_headers(self):
        cfg = parse_intercept_headers({})
        assert cfg.multimodal == "auto" and cfg.mm_order_hint == "sentence"
        cfg = parse_intercept_headers(
            {"X-ContextPilot-Multimodal": "off", "X-ContextPilot-MM-Order-Hint": "none"}
        )
        assert cfg.multimodal == "off" and cfg.mm_order_hint == "none"
        cfg = parse_intercept_headers({"x-contextpilot-mm-order-hint": "weird"})
        assert cfg.mm_order_hint == "sentence"


class TestServerIntegration:
    """Exercise the http_server multimodal branch without a backend."""

    def test_intercept_multimodal_reorders_second_request(self, monkeypatch):
        from contextpilot.server import http_server as hs
        from contextpilot.server.intercept_parser import InterceptConfig

        monkeypatch.setattr(hs, "_intercept_index", None)
        cfg = InterceptConfig()

        def _permute(body, perm):
            """Reorder (label, image) pairs of the user message by perm."""
            c = body["messages"][1]["content"]
            pairs = [(c[i], c[i + 1]) for i in range(1, len(c) - 1, 2)]
            new = [c[0]] + [x for j in perm for x in pairs[j]] + [c[-1]]
            body["messages"][1]["content"] = new
            return body

        # First request arrives in retrieval order [3, 1, 4, 2] (not chronological)
        body1 = _permute(_mm_body(4), [2, 0, 3, 1])
        out1, d1 = hs._intercept_multimodal(body1, cfg)
        assert d1["count"] == 4 and d1["original_order"] == d1["reordered_order"]
        assert d1["order_from_labels"]
        t1 = out1["messages"][1]["content"][-1]["text"]
        assert t1.startswith("Note: the frames above are NOT")
        assert "[Frame 1 | t=0.0s], [Frame 2 | t=2.0s], [Frame 3 | t=4.0s], [Frame 4 | t=6.0s]" in t1

        # Second request: frames 1..4 again plus a new frame 5 in front
        body2 = _permute(_mm_body(5), [4, 1, 3, 0, 2])
        out2, d2 = hs._intercept_multimodal(body2, cfg)
        assert d2["count"] == 5
        assert d2["original_order"] != d2["reordered_order"]
        content = out2["messages"][1]["content"]
        imgs = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
        first_imgs = [
            p["image_url"]["url"]
            for p in body1["messages"][1]["content"]
            if p["type"] == "image_url"
        ]
        assert imgs[:4] == first_imgs  # cached frames first (same order), new frame last
        # labels travel with their images
        lbls = [p["text"] for p in content[1:-1] if p["type"] == "text"]
        assert lbls[:4] == [
            p["text"] for p in body1["messages"][1]["content"][1:-1] if p["type"] == "text"
        ]
        assert "true chronological order" in content[-1]["text"]
        assert content[-1]["text"].endswith("Which happens first?")


# ── canonical prefixes ───────────────────────────────────────────────────


from contextpilot.multimodal import (  # noqa: E402
    group_execution_order,
    plan_canonical,
    plan_canonical_batch,
    select_core,
)


class TestCanonicalPrefix:
    def test_select_core_by_share(self):
        f = [_blk(i) for i in range(6)]
        lists = [[f[0], f[1], f[4]], [f[1], f[0], f[5]], [f[0], f[2], f[3]]]
        core = select_core(lists, min_share=0.5)          # >= 2 of 3 requests
        assert [b.key for b in core] == [f[0].key, f[1].key]
        core = select_core(lists, min_share=1.0)          # all 3 requests
        assert [b.key for b in core] == [f[0].key]

    def test_select_core_is_chronological(self):
        f = [_blk(i) for i in range(4)]
        lists = [[f[3], f[1]], [f[1], f[3]]]
        assert [b.order for b in select_core(lists, min_share=1.0)] == [1.0, 3.0]

    def test_select_core_size_cap_and_min_core(self):
        f = [_blk(i) for i in range(5)]
        lists = [[f[0], f[1], f[2]], [f[0], f[1], f[3]], [f[0], f[4], f[1]]]
        core = select_core(lists, min_share=1.0, core_size=2)
        assert len(core) == 2
        # no frame shared by all → min_core still returns something
        lists = [[f[0]], [f[1]], [f[2]]]
        assert len(select_core(lists, min_share=1.0, min_core=1)) == 1
        assert select_core([], min_share=1.0) == []

    def test_plan_canonical_prefix_then_tail(self):
        f = [_blk(i) for i in range(6)]
        core = [f[1], f[2]]
        out = plan_canonical([f[4], f[2], f[0]], core, include_missing=True)
        # core first (in core order, incl. the frame not retrieved), then tail
        assert [b.key for b in out] == [f[1].key, f[2].key, f[0].key, f[4].key]

    def test_plan_canonical_without_missing(self):
        f = [_blk(i) for i in range(6)]
        out = plan_canonical([f[4], f[2], f[0]], [f[1], f[2]], include_missing=False)
        assert [b.key for b in out] == [f[2].key, f[0].key, f[4].key]

    def test_plan_canonical_tail_order(self):
        f = [_blk(i) for i in range(6)]
        out = plan_canonical([f[5], f[3]], [f[1]], include_missing=True, tail_chronological=False)
        assert [b.key for b in out] == [f[1].key, f[5].key, f[3].key]

    def test_batch_gives_one_identical_boundary_per_group(self):
        f = [_blk(i) for i in range(10)]
        reqs = [
            [f[0], f[1], f[5]],          # video A
            [f[1], f[0], f[6]],          # video A
            [f[0], f[7]],                # video A
            [f[2], f[3]],                # video B
            [f[3], f[2], f[8]],          # video B
        ]
        groups = ["A", "A", "A", "B", "B"]
        displayed, cores = plan_canonical_batch(reqs, groups, min_share=0.5)
        assert [b.key for b in cores["A"]] == [f[0].key, f[1].key]
        assert [b.key for b in cores["B"]] == [f[2].key, f[3].key]
        # every request of a group starts with that group's full core
        for d, g in zip(displayed, groups):
            assert [b.key for b in d[: len(cores[g])]] == [b.key for b in cores[g]]
        # each request still shows all of its own frames
        for d, r in zip(displayed, reqs):
            assert {b.key for b in r} <= {b.key for b in d}

    def test_batch_length_mismatch(self):
        with pytest.raises(ValueError):
            plan_canonical_batch([[_blk(0)]], ["a", "b"])

    def test_group_execution_order(self):
        assert group_execution_order(["a", "b", "a", "c", "b"]) == [0, 2, 1, 4, 3]
        assert group_execution_order([]) == []

    def test_canonical_messages_share_a_prefix(self):
        """The assembled prompts of one group share their leading parts."""
        f = [_blk(i) for i in range(8)]
        # core frames (5, 6) are LATER in time than the per-request frames,
        # so the canonical display order is not chronological
        reqs = [[f[5], f[6], f[0]], [f[6], f[5], f[1]], [f[5], f[6], f[2]]]
        displayed, cores = plan_canonical_batch(reqs, ["v", "v", "v"], min_share=0.5)
        assert [b.key for b in cores["v"]] == [f[5].key, f[6].key]
        msgs = [build_multimodal_messages(d, f"q{i}?", order_hint="sentence")
                for i, d in enumerate(displayed)]
        contents = [m[-1]["content"] for m in msgs]
        # intro + 2 core images identical across all three prompts
        assert contents[0][:3] == contents[1][:3] == contents[2][:3]
        # only the trailing text (hint + question) differs
        assert contents[0][-1] != contents[1][-1]
        # and the hint states the true chronological order
        tail = contents[0][-1]["text"]
        assert tail.startswith("Note: the frames above are NOT")
        assert "[Frame 1], [Frame 6], [Frame 7]" in tail
        assert tail.endswith("q0?")

    def test_budget_keeps_frame_count_constant(self):
        f = [_blk(i) for i in range(10)]
        core = [f[7], f[8]]           # neither retrieved by this request
        blocks = [f[0], f[1], f[2], f[3]]   # retrieval order, best first
        out = plan_canonical(blocks, core, include_missing=True, budget=4)
        assert len(out) == 4
        assert [b.key for b in out[:2]] == [f[7].key, f[8].key]
        # lowest-ranked own frames dropped, survivors chronological
        assert [b.key for b in out[2:]] == [f[0].key, f[1].key]

    def test_budget_never_truncates_the_core(self):
        f = [_blk(i) for i in range(6)]
        out = plan_canonical([f[0]], [f[3], f[4], f[5]], include_missing=True, budget=2)
        assert [b.key for b in out] == [f[3].key, f[4].key, f[5].key]

    def test_budget_in_batch_equalizes_length(self):
        f = [_blk(i) for i in range(12)]
        reqs = [[f[0], f[1], f[9], f[10]], [f[1], f[0], f[8], f[11]], [f[0], f[1], f[7], f[6]]]
        displayed, cores = plan_canonical_batch(
            reqs, ["v"] * 3, min_share=0.5, budget=4
        )
        assert all(len(d) == 4 for d in displayed)
        n = len(cores["v"])
        assert all([b.key for b in d[:n]] == [b.key for b in cores["v"]] for d in displayed)
