"""Demo end-to-end: doc video -> detection -> classification -> overlay + CSV.

Chay:
    python src/demo.py                       # xuat video + CSV vao outputs/
    python src/demo.py --show                # xem truc tiep (phim q de thoat)
    python src/demo.py --video other.mp4
"""

import argparse
import csv
import os
import time
from collections import Counter

import cv2

import config as cfg
from classifier import classify
from detector import Detector
from tracker import CentroidTracker


def draw_overlay(frame, dets, line_x, counts, fps, frame_idx):
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

    for d in dets:
        x, y, w, h = d["bbox"]
        color = cfg.DRAW_COLORS.get(d["shape"], cfg.DRAW_COLORS["unknown"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, d["centroid"], 3, color, -1)
        tag = f"#{d.get('id', '-')} {d['label']} {d['size']}"
        cv2.putText(frame, tag, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Bang thong tin
    lines = [f"frame {frame_idx}  |  {fps:5.1f} FPS  |  detected: {len(dets)}"]
    lines += [f"{k:>18}: {v}" for k, v in sorted(counts.items())]
    lines += [f"{'TOTAL':>18}: {sum(counts.values())}"]
    box_h = 18 * len(lines) + 10
    cv2.rectangle(frame, (5, 5), (330, box_h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 24 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=cfg.VIDEO_PATH)
    ap.add_argument("--out", default=cfg.OUT_VIDEO)
    ap.add_argument("--csv", default=cfg.OUT_CSV)
    ap.add_argument("--show", action="store_true", help="hien cua so xem truc tiep")
    ap.add_argument("--mask", action="store_true", help="ghi them video mask tach nen")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    line_x = int(W * cfg.COUNT_LINE_RATIO)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, src_fps, (W, H))
    mask_writer = (cv2.VideoWriter(args.out.replace(".mp4", "_mask.mp4"),
                                   fourcc, src_fps, (W, H), False)
                   if args.mask else None)

    detector = Detector()
    tracker = CentroidTracker()
    counts = Counter()
    rows = []
    rejected = []   # blob khong khop primitive nao -> khong dem, ghi rieng de QA

    frame_idx = 0
    t_start = time.perf_counter()
    proc_time = 0.0
    fps_disp = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        dets, mask, hsv = detector.detect(frame)   # hsv tai su dung, khong tinh lai
        for d in dets:
            classify(d, hsv, frame.shape)
        tracker.update(dets)
        for d in dets:
            if not tracker.check_line_crossing(d, line_x):
                continue
            if d["shape"] == "unknown":
                # Thuong la hai san pham cung mau cham nhau bi gop thanh 1 contour.
                # Khong dem va khong bia nhan; ghi ra file rieng de kiem tra.
                rejected.append({
                    "timestamp_s": round(frame_idx / src_fps, 3),
                    "frame": frame_idx,
                    "id": d["id"],
                    "color": d["color"],
                    "area_px": int(d["area"]),
                    "cx": d["centroid"][0],
                    "cy": d["centroid"][1],
                    "best_fit": d["shape_conf"],
                    **{f"fit_{k}": v for k, v in d["features"]["fit_scores"].items()},
                })
            else:
                counts[d["label"]] += 1
                rows.append({
                    "timestamp_s": round(frame_idx / src_fps, 3),
                    "frame": frame_idx,
                    "id": d["id"],
                    "shape": d["shape"],
                    "color": d["color"],
                    "size": d["size"],
                    "label": d["label"],
                    "area_px": int(d["area"]),
                    "cx": d["centroid"][0],
                    "cy": d["centroid"][1],
                    "shape_conf": d["shape_conf"],
                    "count_total": sum(counts.values()),
                })
        dt = time.perf_counter() - t0
        proc_time += dt
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / dt) if fps_disp else 1.0 / dt

        out = draw_overlay(frame, dets, line_x, counts, fps_disp, frame_idx)
        writer.write(out)
        if mask_writer is not None:
            mask_writer.write(mask)
        if args.show:
            cv2.imshow("MVP1 - Detection & Classification", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["timestamp_s", "frame", "id", "shape", "color",
                            "size", "label", "area_px", "cx", "cy",
                            "shape_conf", "count_total"])
        w.writeheader()
        w.writerows(rows)

    rej_path = args.csv.replace(".csv", "_rejected.csv")
    if rejected:
        with open(rej_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rejected[0].keys()))
            w.writeheader()
            w.writerows(rejected)

    wall = time.perf_counter() - t_start
    print(f"Frames xu ly      : {frame_idx}")
    print(f"FPS xu ly (thuat toan): {frame_idx / proc_time:.1f}")
    print(f"FPS ke ca I/O ghi video: {frame_idx / wall:.1f}")
    print(f"Tong san pham dem duoc : {sum(counts.values())}")
    for k, v in sorted(counts.items()):
        print(f"  {k:>18}: {v}")
    if rejected:
        print(f"Blob bi tu choi (khong khop hinh nao): {len(rejected)}")
        for r in rejected:
            print(f"  frame {r['frame']:>4}  #{r['id']:<3} {r['color']:<8}"
                  f" best_fit={r['best_fit']:.3f}")
    print(f"Video  -> {args.out}")
    print(f"CSV    -> {args.csv}")
    if rejected:
        print(f"Reject -> {rej_path}")


if __name__ == "__main__":
    main()
