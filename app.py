"""Web demo cho pipeline MVP1 — Nhóm 2 (Detection & Classification).

Chạy local:  streamlit run app.py
Deploy:      xem render.yaml
"""

import csv
import io
import os
import sys
import tempfile
import time
from collections import Counter

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as cfg
from classifier import classify
from demo import draw_overlay
from detector import Detector
from tracker import CentroidTracker

st.set_page_config(
    page_title="MVP1 — Detection & Classification",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)

SAMPLE_VIDEO = os.path.join("data", "conveyor_2d.mp4")

CSV_FIELDS = [
    "timestamp_s", "frame", "id", "shape", "color", "size",
    "label", "area_px", "cx", "cy", "shape_conf", "count_total",
]


def process(video_path, params, preview_slot, progress, preview_every):
    """Chạy pipeline, đẩy frame xem trước ra UI, trả về kết quả."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    line_x = int(W * params["line_ratio"])

    out_path = os.path.join(tempfile.mkdtemp(), "annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))

    detector = Detector(
        sat_min=params["sat_min"],
        val_min=params["val_min"],
        min_area=params["min_area"],
    )
    tracker = CentroidTracker()
    counts = Counter()
    rows = []
    frame_idx = 0
    proc_time = 0.0
    fps_disp = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        dets, mask, hsv = detector.detect(frame)
        for d in dets:
            classify(d, hsv, frame.shape)
        tracker.update(dets)
        for d in dets:
            if tracker.check_line_crossing(d, line_x):
                counts[d["label"]] += 1
                rows.append({
                    "timestamp_s": round(frame_idx / fps_src, 3),
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
        fps_disp = 0.9 * fps_disp + 0.1 / dt if fps_disp else 1.0 / dt

        annotated = draw_overlay(frame, dets, line_x, counts, fps_disp, frame_idx)
        writer.write(annotated)

        # Chỉ đẩy ảnh xem trước mỗi N frame -> đỡ tốn băng thông khi chạy trên server
        if frame_idx % preview_every == 0:
            shown = mask if params["show_mask"] else cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_slot.image(shown, width="stretch")
            if total:
                progress.progress(min(frame_idx / total, 1.0),
                                  text=f"Frame {frame_idx}/{total}")

    cap.release()
    writer.release()

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    w.writerows(rows)

    with open(out_path, "rb") as f:
        video_bytes = f.read()

    return {
        "frames": frame_idx,
        "fps_algo": frame_idx / proc_time if proc_time else 0.0,
        "counts": dict(counts),
        "rows": rows,
        "csv": buf.getvalue(),
        "video": video_bytes,
    }


st.title("Nhận diện và phân loại sản phẩm trên băng chuyền")
st.caption(
    "MVP1 — Nhóm 2 · Pipeline OpenCV classical: tách nền → contour → đặc trưng hình học "
    "→ phân loại hình dạng/màu/kích thước → đếm qua vạch."
)

with st.sidebar:
    st.header("Nguồn video")
    source = st.segmented_control(
        "Nguồn", ["Video mẫu", "Tải lên"],
        default="Video mẫu", label_visibility="collapsed",
    )
    uploaded = None
    if source == "Tải lên":
        uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])

    st.header("Ngưỡng tách nền")
    sat_min = st.slider(
        "Bão hoà tối thiểu (SAT_MIN)", 0, 255, cfg.SAT_MIN,
        help="Tăng nếu nền lọt vào mask; giảm nếu sản phẩm bị thủng lỗ.",
    )
    val_min = st.slider("Độ sáng tối thiểu (VAL_MIN)", 0, 255, cfg.VAL_MIN)
    min_area = st.slider(
        "Diện tích tối thiểu (MIN_AREA)", 50, 5000, cfg.MIN_AREA, step=50,
        help="Contour nhỏ hơn ngưỡng này bị coi là nhiễu.",
    )
    line_ratio = st.slider("Vị trí vạch đếm", 0.05, 0.95, cfg.COUNT_LINE_RATIO, step=0.05)

    st.header("Hiển thị")
    show_mask = st.toggle("Xem mask tách nền", help="Dùng để chẩn đoán khi tune ngưỡng.")
    preview_every = st.slider(
        "Cập nhật xem trước mỗi N frame", 1, 30, 3,
        help="Tăng lên nếu server yếu, để đỡ tốn băng thông đẩy ảnh.",
    )

run = st.button("Chạy pipeline", icon=":material/play_arrow:", type="primary")

preview_slot = st.empty()
progress_slot = st.empty()

if run:
    if source == "Tải lên":
        if uploaded is None:
            st.warning("Hãy tải lên một video trước.", icon=":material/upload:")
            st.stop()
        suffix = os.path.splitext(uploaded.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()
        path = tmp.name
    else:
        if not os.path.exists(SAMPLE_VIDEO):
            st.error(f"Không tìm thấy video mẫu tại `{SAMPLE_VIDEO}`.")
            st.stop()
        path = SAMPLE_VIDEO

    params = {
        "sat_min": sat_min,
        "val_min": val_min,
        "min_area": min_area,
        "line_ratio": line_ratio,
        "show_mask": show_mask,
    }
    with st.spinner("Đang xử lý video..."):
        st.session_state["result"] = process(
            path, params, preview_slot, progress_slot, preview_every
        )
    progress_slot.empty()

result = st.session_state.get("result")

if result:
    kpi_fps = result["fps_algo"]
    with st.container(horizontal=True):
        st.metric("Frame đã xử lý", result["frames"])
        st.metric("FPS thuật toán", f"{kpi_fps:.1f}",
                  delta=f"{kpi_fps - 30:+.1f} so với KPI")
        st.metric("Sản phẩm đếm được", sum(result["counts"].values()))
        st.metric("Số loại nhãn", len(result["counts"]))

    if kpi_fps < 30:
        st.warning(
            "FPS dưới KPI 30 — server demo yếu hơn máy local. "
            "Đo KPI nghiệm thu trên máy thật, không lấy con số ở đây.",
            icon=":material/warning:",
        )

    st.subheader("Phân bố nhãn")
    dist = (
        pd.Series(result["counts"], name="count")
        .rename_axis("label")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    st.bar_chart(dist, x="label", y="count", horizontal=True, height=380)

    st.subheader("Nhật ký đếm (log.csv)")
    st.caption("Mỗi dòng là một sản phẩm vượt vạch đếm. Mỗi ID chỉ được đếm đúng một lần.")
    st.dataframe(pd.DataFrame(result["rows"], columns=CSV_FIELDS), height=320, key="log_table")

    with st.container(horizontal=True):
        st.download_button(
            "Tải log.csv", result["csv"], "log.csv", "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Tải video đã gán nhãn", result["video"], "demo_annotated.mp4", "video/mp4",
            icon=":material/download:",
        )
    st.caption(
        "Video xuất bằng codec mp4v nên trình duyệt thường không phát trực tiếp — "
        "tải về mở bằng VLC, hoặc xem trực tiếp ở khung xem trước phía trên."
    )
else:
    st.info("Chọn nguồn video ở thanh bên rồi bấm **Chạy pipeline**.", icon=":material/info:")
