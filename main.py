#!/usr/bin/env python3
"""assets/ 配下の setlog-MMDD.mp4 を日付順に結合して output.mp4 を作る。"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
OUTPUT = Path(__file__).parent / "output.mp4"
FACE_TXT = Path(__file__).parent / "face.txt"
BOUNDARIES_DIR = Path(__file__).parent / "boundaries"
# --boundary で検出境界フレームの前後に書き出すフレーム数（±N）。
BOUNDARY_CONTEXT_FRAMES = 2
YUNET_MODEL = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
PATTERN = re.compile(r"setlog-(\d{4})\.mp4$")
SPEED = 3.0
# 各セグメントの公称秒数（素材が n×NOMINAL_SEGMENT_SEC で構成されている前提）。
NOMINAL_SEGMENT_SEC = 2.0
# 各セグメントから抽出するサンプル数。等間隔で抜いて検出に回す。
SAMPLES_PER_SEGMENT = 5
# YuNet の信頼度しきい値。
FACE_SCORE_THRESHOLD = 0.6
# 期待境界の前後何秒を実境界探索窓とするか。
SCENE_SEARCH_WINDOW_SEC = 0.5
# シーン切替検出用のダウンスケール解像度（NxN）。全画面の粗い差分で判定する。
SCENE_DIFF_SIZE = 32
# boxblur の半径（px）。
BLUR_RADIUS = 35
# --test で重畳するテキスト用フォントと生成 PNG 置き場。
TEST_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
TEST_LABEL_DIR = Path(__file__).parent / ".labels"
TEST_FONT_SIZE = 28


def collect() -> list[Path]:
    files = []
    for p in ASSETS.iterdir():
        m = PATTERN.match(p.name)
        if m:
            files.append((m.group(1), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def scene_segments(clip: Path) -> tuple[list[tuple[float, float]], float]:
    """全画面ダウンスケール差分で切替フレームを拾い、(start,end) 秒のリストと fps を返す。"""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total <= 0:
        cap.release()
        return [], 0.0

    duration = total / fps

    # タイマー ROI だけだと素材内の動き（手や人物が通過する動きブラー）で偽ピークが立つため、
    # 画面全体をダウンスケールしたフレーム差分でシーン切替を検出する。
    diffs = [0] * total
    prev = None
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SCENE_DIFF_SIZE, SCENE_DIFF_SIZE))
        if prev is not None:
            diffs[idx] = int(cv2.absdiff(small, prev).sum())
        prev = small
    cap.release()

    # 公称セグメント数 n を推定し、各期待境界フレーム近傍で diff 最大のフレームを採用。
    n = max(1, int(round(duration / NOMINAL_SEGMENT_SEC)))
    window = max(1, int(SCENE_SEARCH_WINDOW_SEC * fps))
    boundaries: list[int] = []
    for k in range(1, n):
        expected = int(round(k * total / n))
        lo = max(1, expected - window)
        hi = min(total, expected + window + 1)
        best = max(range(lo, hi), key=lambda i: diffs[i])
        boundaries.append(best)

    cuts = [0, *boundaries, total]
    segments = [(cuts[i] / fps, cuts[i + 1] / fps) for i in range(len(cuts) - 1)]
    return segments, fps


def load_blur_map() -> dict[str, list[int]]:
    if not FACE_TXT.exists():
        print(f"{FACE_TXT.name} not found; run --face first", file=sys.stderr)
        return {}
    mapping: dict[str, list[int]] = {}
    for line in FACE_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "/" not in line:
            continue
        name, _, idx = line.rpartition("/")
        try:
            mapping.setdefault(name, []).append(int(idx))
        except ValueError:
            continue
    return mapping


def make_label_png(text: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    TEST_LABEL_DIR.mkdir(exist_ok=True)
    safe = text.replace("/", "_")
    out = TEST_LABEL_DIR / f"{safe}.png"
    if out.exists():
        return out
    font = ImageFont.truetype(TEST_FONT, TEST_FONT_SIZE)
    pad = 10
    tmp = Image.new("RGBA", (10, 10))
    tb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 160))
    ImageDraw.Draw(img).text((pad - tb[0], pad - tb[1]), text, font=font, fill=(255, 255, 255, 255))
    img.save(out)
    return out


def main(blur: bool = False, test: bool = False) -> int:
    clips = collect()
    if not clips:
        print("no clips found in assets/", file=sys.stderr)
        return 1

    blur_map = load_blur_map() if blur else {}

    inputs: list[str] = []
    for p in clips:
        inputs += ["-i", str(p)]

    # --test 用のラベル PNG を追加入力として積む（clip_index, seg_index -> ffmpeg 入力 index）。
    label_inputs: dict[tuple[int, int], int] = {}

    pre_filters: list[str] = []
    concat_labels: list[str] = []
    for i, p in enumerate(clips):
        src = f"[{i}:v:0]"
        segs = blur_map.get(p.name, []) if blur else []
        need_segs = bool(segs) or (test and blur)
        ranges_sec: list[tuple[float, float]] = []
        fps = 0.0
        if need_segs:
            ranges_sec, fps = scene_segments(p)

        cur = src
        if segs:
            # 検出境界が1フレーム程度ずれる余地があるので前後1フレーム分広げてぼかす。
            pad = (1.0 / fps) if fps > 0 else 0.0
            parts_r = [
                f"between(t,{max(0.0, ranges_sec[n-1][0] - pad):.6f},{ranges_sec[n-1][1] + pad:.6f})"
                for n in segs if 1 <= n <= len(ranges_sec)
            ]
            if parts_r:
                nxt = f"[b{i}]"
                pre_filters.append(f"{cur}boxblur={BLUR_RADIUS}:enable='{'+'.join(parts_r)}'{nxt}")
                cur = nxt

        if test and blur and ranges_sec:
            for k, (s, e) in enumerate(ranges_sec, start=1):
                png = make_label_png(f"{p.name}/{k}")
                in_idx = len(clips) + len(label_inputs)
                label_inputs[(i, k)] = in_idx
                inputs += ["-i", str(png)]
                nxt = f"[t{i}_{k}]"
                pre_filters.append(
                    f"{cur}[{in_idx}:v]overlay=W-w-20:20:enable='between(t,{s:.6f},{e:.6f})'{nxt}"
                )
                cur = nxt

        concat_labels.append(cur)

    concat_in = "".join(concat_labels)
    parts = list(pre_filters)
    parts.append(f"{concat_in}concat=n={len(clips)}:v=1:a=0[cv]")
    if SPEED != 1.0:
        parts.append(f"[cv]setpts=PTS/{SPEED}[v]")
        out_label = "[v]"
    else:
        out_label = "[cv]"
    filtergraph = ";".join(parts)
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filtergraph,
        "-map", out_label,
        "-an",
        str(OUTPUT),
    ]
    print("concatenating:", *[p.name for p in clips], "(blur)" if blur else "")
    result = subprocess.run(cmd)
    return result.returncode


def detect_faces() -> int:
    import cv2  # noqa: PLC0415

    if not YUNET_MODEL.exists():
        print(f"model not found: {YUNET_MODEL}", file=sys.stderr)
        return 1

    clips = collect()
    if not clips:
        print("no clips found in assets/", file=sys.stderr)
        return 1

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (320, 320),
        score_threshold=FACE_SCORE_THRESHOLD,
        nms_threshold=0.3, top_k=50,
    )

    entries: list[str] = []
    for clip in clips:
        segments, fps = scene_segments(clip)
        if not segments or fps <= 0:
            print(f"invalid metadata for {clip.name}", file=sys.stderr)
            continue
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"cannot open {clip.name}", file=sys.stderr)
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        detector.setInputSize((w, h))
        hits: list[int] = []
        for seg_idx, (start, end) in enumerate(segments, start=1):
            found = False
            for i in range(SAMPLES_PER_SEGMENT):
                t = start + (end - start) * (i + 0.5) / SAMPLES_PER_SEGMENT
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, frame = cap.read()
                if not ok:
                    continue
                _, faces = detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    found = True
                    break
            if found:
                hits.append(seg_idx)
        cap.release()
        print(f"{clip.name}: {len(segments)} segments, hits={hits}")
        for n in hits:
            entries.append(f"{clip.name}/{n}")

    FACE_TXT.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    print(f"wrote {len(entries)} entries to {FACE_TXT.name}")
    return 0


def dump_boundaries() -> int:
    """各クリップの検出境界フレーム±BOUNDARY_CONTEXT_FRAMES を横並びにして1枚ずつ書き出す。"""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    clips = collect()
    if not clips:
        print("no clips found in assets/", file=sys.stderr)
        return 1

    blur_map = load_blur_map()
    if not blur_map:
        return 1

    BOUNDARIES_DIR.mkdir(exist_ok=True)
    offsets = list(range(-BOUNDARY_CONTEXT_FRAMES, BOUNDARY_CONTEXT_FRAMES + 1))
    for clip in clips:
        blur_segs = set(blur_map.get(clip.name, []))
        if not blur_segs:
            continue
        segments, fps = scene_segments(clip)
        if not segments or fps <= 0:
            print(f"invalid metadata for {clip.name}", file=sys.stderr)
            continue
        boundary_frames = [int(round(s * fps)) for s, _ in segments[1:]]
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"cannot open {clip.name}", file=sys.stderr)
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dumped = 0
        for k, bf in enumerate(boundary_frames, start=2):
            # セグメント (k-1) と k の境界。少なくとも片方がぼかし対象のときだけ書き出す。
            if (k - 1) not in blur_segs and k not in blur_segs:
                continue
            dumped += 1
            tiles = []
            for off in offsets:
                fr = bf + off
                if 0 <= fr < total:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ok, frame = cap.read()
                    tile = frame if ok else np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    tile = np.zeros((h, w, 3), dtype=np.uint8)
                # 境界フレーム (off=0) は赤枠、他は白枠で見分けやすく。
                color = (0, 0, 255) if off == 0 else (255, 255, 255)
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 6)
                cv2.putText(tile, f"{off:+d}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
                tiles.append(tile)
            strip = np.hstack(tiles)
            out = BOUNDARIES_DIR / f"{clip.stem}_seg{k}.png"
            cv2.imwrite(str(out), strip)
        cap.release()
        print(f"{clip.name}: dumped {dumped} boundaries")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--face" in args:
        sys.exit(detect_faces())
    if "--boundary" in args:
        sys.exit(dump_boundaries())
    sys.exit(main(blur="--blur" in args, test="--test" in args))
