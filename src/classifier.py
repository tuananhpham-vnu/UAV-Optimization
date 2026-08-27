"""Classification: trich xuat dac trung tu contour -> gan nhan hinh dang / mau / kich thuoc.

Dac trung dung (deu bat bien voi phep tinh tien, khong can training):
  - circularity = 4*pi*A / P^2        : tron ~1.00, vuong ~0.79, tam giac ~0.60
  - extent      = A / A_bbox          : vuong ~1.00, tron ~0.79, tam giac ~0.50
  - n_vertices  = so dinh sau approxPolyDP
  - hue trung vi ben trong contour    : dinh danh mau
  - area                              : dinh danh kich thuoc S/M/L
"""

import cv2
import numpy as np

import config as cfg

_ERODE_K = np.ones((3, 3), np.uint8)


def extract_features(det, hsv):
    """Tinh vector dac trung hinh hoc + mau cho mot detection."""
    cnt = det["contour"]
    area = det["area"]
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = det["bbox"]

    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    extent = area / float(w * h) if w * h > 0 else 0.0
    aspect = w / float(h) if h > 0 else 0.0

    approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
    n_vertices = len(approx)

    # Hue trung vi -> khong bi keo lech boi vien khu rang cua.
    # Chi to mask trong pham vi bounding box (ROI) de khong cap phat full-frame
    # cho tung vat -> nhanh hon nhieu lan khi khung hinh co ~20 vat.
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi_mask, [cnt - (x, y)], -1, 255, cv2.FILLED)
    roi_mask = cv2.erode(roi_mask, _ERODE_K)  # bo vien pha mau
    px = hsv[y:y + h, x:x + w, 0][roi_mask == 255]
    hue = float(np.median(px)) if px.size else -1.0

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "extent": extent,
        "aspect": aspect,
        "n_vertices": n_vertices,
        "hue": hue,
    }


def shape_fit_scores(cnt, area):
    """Cham diem contour khop voi tung primitive: A_contour / A_primitive.

    Khop hoan hao -> 1.0. Uu diem so voi circularity/extent don thuan: blob do
    nhieu vat cham nhau khong khop primitive nao, nen bi phat hien thay vi bi
    ep vao nhan gan nhat.
    """
    _, radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * radius * radius

    (_, (rw, rh), _) = cv2.minAreaRect(cnt)
    rect_area = rw * rh

    tri_area, _ = cv2.minEnclosingTriangle(cnt)

    return {
        "circle": area / circle_area if circle_area > 0 else 0.0,
        "square": area / rect_area if rect_area > 0 else 0.0,
        "triangle": area / float(tri_area) if tri_area > 0 else 0.0,
    }


def classify_shape(cnt, area):
    """Gan nhan hinh dang + do tin cay = diem khop.

    Tra ve ("unknown", diem) khi khong primitive nao khop du tot — thuong la
    hai san pham cung mau cham nhau bi gop thanh mot contour. Tra "unknown"
    tot hon la bia ra mot nhan trong nhu chac chan.
    """
    scores = shape_fit_scores(cnt, area)
    label = max(scores, key=scores.get)
    best = scores[label]
    if best < cfg.SHAPE_FIT_MIN:
        return "unknown", best, scores
    return label, best, scores


def classify_color(f):
    hue = f["hue"]
    if hue < 0:
        return "unknown"
    for name, ranges in cfg.COLOR_RANGES:
        for lo, hi in ranges:
            if lo <= hue <= hi:
                return name
    return "unknown"


def classify_size(f):
    a = f["area"]
    if a <= cfg.SIZE_SMALL_MAX:
        return "S"
    if a <= cfg.SIZE_MEDIUM_MAX:
        return "M"
    return "L"


def classify(det, hsv, frame_shape):
    """Gan nhan day du cho mot detection, ghi thang vao dict det."""
    f = extract_features(det, hsv)
    x, y, w, h = det["bbox"]
    H, W = frame_shape[:2]
    # Vat bi cat o mep khung -> dac trung hinh hoc khong dang tin
    truncated = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1

    shape, conf, scores = classify_shape(det["contour"], det["area"])
    f["fit_scores"] = {k: round(v, 3) for k, v in scores.items()}
    det.update(
        {
            "features": f,
            "shape": "unknown" if truncated else shape,
            "shape_conf": 0.0 if truncated else round(conf, 3),
            # color_hint den tu buoc tach mask theo mau; hue trung vi la kiem chung
            "color": det.get("color_hint") or classify_color(f),
            "size": classify_size(f),
            "truncated": truncated,
        }
    )
    det["label"] = f"{det['color']}_{det['shape']}"
    return det
