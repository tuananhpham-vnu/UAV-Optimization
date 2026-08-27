"""Classification: trich xuat dac trung tu contour -> gan nhan hinh dang / mau / kich thuoc.

Dac trung dung (deu bat bien voi phep tinh tien, khong can training):
  - circularity = 4*pi*A / P^2        : tron ~1.00, vuong ~0.79, tam giac ~0.60
  - extent      = A / A_bbox          : vuong ~1.00, tron ~0.79, tam giac ~0.50
  - n_vertices  = so dinh sau approxPolyDP
  - hue trung vi ben trong contour    : dinh danh mau
  - area                              : dinh danh kich thuoc S/M/L
"""

import itertools

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


def _triangle_of(cnt):
    """Tam giac khop nhat voi contour: rut convex hull ve dung 3 dinh.

    Khong dung cv2.minEnclosingTriangle: ham do tra ve ket qua sai tren mot so
    point set (da gap tam giac dien tich 5811 px nhung bao boi tam giac 8694 px,
    dinh nam ngoai han hull), khien tam giac that bi cham diem nhu blob hong.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    # Tim epsilon nho nhat cho ra <= 3 dinh
    lo, hi = 0.01, 0.5
    for _ in range(20):
        mid = (lo + hi) / 2
        if len(cv2.approxPolyDP(hull, mid * peri, True)) > 3:
            lo = mid
        else:
            hi = mid
    approx = cv2.approxPolyDP(hull, hi * peri, True).reshape(-1, 2)
    if len(approx) == 3:
        return approx.astype(np.int32)
    # approxPolyDP suy bien -> chon 3 dinh hull cho dien tich lon nhat
    pts = hull.reshape(-1, 2)
    if len(pts) < 3:
        return None
    best, best_area = None, -1.0
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        tri = pts[[i, j, k]]
        ar = abs(cv2.contourArea(tri.astype(np.int32)))
        if ar > best_area:
            best, best_area = tri, ar
    return best.astype(np.int32) if best is not None else None


def shape_fit_scores(cnt, area=None):
    """Cham diem contour khop voi tung primitive bang IoU (intersection over union).

    IoU doi xung: phat ca truong hop primitive to hon vat lan nho hon vat, nen
    blob do nhieu vat cham nhau khong the an diem cao o bat ky primitive nao.
    Khop hoan hao -> 1.0.
    """
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    rect = cv2.boxPoints(cv2.minAreaRect(cnt))
    tri = _triangle_of(cnt)

    # Canvas du rong cho ca contour lan moi primitive
    bounds = [cnt.reshape(-1, 2), rect,
              np.array([[cx - radius, cy - radius], [cx + radius, cy + radius]])]
    if tri is not None:
        bounds.append(tri)
    pts = np.vstack(bounds).astype(np.float64)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    h, w = int(y1 - y0), int(x1 - x0)
    if h <= 0 or w <= 0:
        return {"circle": 0.0, "square": 0.0, "triangle": 0.0}

    obj = np.zeros((h, w), np.uint8)
    cv2.drawContours(obj, [cnt - (x0, y0)], -1, 255, cv2.FILLED)

    def iou(draw):
        prim = np.zeros((h, w), np.uint8)
        draw(prim)
        inter = cv2.countNonZero(cv2.bitwise_and(obj, prim))
        union = cv2.countNonZero(cv2.bitwise_or(obj, prim))
        return inter / union if union else 0.0

    scores = {
        "circle": iou(lambda m: cv2.circle(
            m, (int(round(cx - x0)), int(round(cy - y0))), int(round(radius)), 255, -1)),
        "square": iou(lambda m: cv2.fillPoly(m, [np.int32(rect - (x0, y0))], 255)),
        "triangle": 0.0 if tri is None else iou(
            lambda m: cv2.fillPoly(m, [np.int32(tri - (x0, y0))], 255)),
    }
    return scores


def classify_shape(cnt, area=None):
    """Gan nhan hinh dang + do tin cay = diem khop.

    Tra ve ("unknown", diem) khi khong primitive nao khop du tot — thuong la
    hai san pham cung mau cham nhau bi gop thanh mot contour. Tra "unknown"
    tot hon la bia ra mot nhan trong nhu chac chan.
    """
    scores = shape_fit_scores(cnt)
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
