"""Core Detection: tien xu ly -> tach nen -> tim contour -> bounding box.

Chien luoc 2 buoc (nhanh + tach duoc vat cham nhau):
  1. Mask bao hoa: bang chuyen xam (S thap) bi loai, san pham mau ruc (S cao) giu lai.
     Chi 1 lan inRange + 1 lan morphology tren toan khung -> re.
  2. Tach mask do theo tung dai Hue: hai san pham khac mau cham nhau se thanh
     hai contour rieng thay vi bi gop lam mot.
  3. Tach hinh hoc (split_touching): hai san pham *cung mau* cham nhau van bi gop
     sau buoc 2 -> cat theo duong noi hai diem lom sau nhat cua contour.

Dau ra la detection tho (chua phan loai) dung chung cho classifier va tracker.
"""

import cv2
import numpy as np

import config as cfg


class Detector:
    def __init__(self, sat_min=None, val_min=None, min_area=None, morph_kernel=None):
        """Cac tham so de None se lay tu config.py.

        Cho phep truyen thang vao thay vi sua bien global trong cfg -> app web
        nhieu nguoi dung cung luc khong dap nguong cua nhau.
        """
        self.sat_min = cfg.SAT_MIN if sat_min is None else sat_min
        self.val_min = cfg.VAL_MIN if val_min is None else val_min
        self.min_area = cfg.MIN_AREA if min_area is None else min_area
        k = cfg.MORPH_KERNEL if morph_kernel is None else morph_kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def segment(self, hsv):
        """Buoc 1: mask nhi phan cua toan bo san pham tren bang chuyen."""
        mask = cv2.inRange(
            hsv,
            np.array([0, self.sat_min, self.val_min]),
            np.array([179, 255, 255]),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def split_by_color(self, hsv, mask):
        """Buoc 2: cat mask tong thanh tung mask theo dai Hue."""
        hue = hsv[:, :, 0]
        out = {}
        for name, ranges in cfg.COLOR_RANGES:
            m = None
            for lo, hi in ranges:
                part = cv2.inRange(hue, np.array(lo, np.uint8), np.array(hi, np.uint8))
                m = part if m is None else cv2.bitwise_or(m, part)
            m = cv2.bitwise_and(m, mask)
            # Khu vien lan mau giua hai vat cham nhau
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._small)
            out[name] = m
        return out

    def detect(self, frame):
        """Tra ve (detections, mask_tong, hsv).

        Moi detection: bbox, centroid, area, contour, color_hint.
        """
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = self.segment(hsv)

        detections = []
        for color_hint, m in self.split_by_color(hsv, mask).items():
            if cv2.countNonZero(m) < self.min_area:
                continue
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for raw in contours:
                if cv2.contourArea(raw) < self.min_area:
                    continue
                # Vat cung mau cham nhau khong tach duoc bang Hue -> tach hinh hoc
                for cnt in split_touching(raw, self.min_area):
                    area = cv2.contourArea(cnt)
                    if area < self.min_area or area > cfg.MAX_AREA:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    mo = cv2.moments(cnt)
                    if mo["m00"] == 0:
                        continue
                    detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "centroid": (int(mo["m10"] / mo["m00"]),
                                         int(mo["m01"] / mo["m00"])),
                            "area": area,
                            "contour": cnt,
                            "color_hint": color_hint,
                        }
                    )
        return detections, mask, hsv


def _deep_defects(cnt):
    """Tra ve 2 diem lom sau nhat cua contour (neu co)."""
    if len(cnt) < 4:
        return None
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None or len(defects) < 2:
        return None
    d = defects.reshape(-1, 4)
    d = d[np.argsort(-d[:, 3])][:2]
    # depth cua OpenCV duoc luu o don vi 1/256 px
    if d[1][3] / 256.0 < cfg.SPLIT_MIN_DEFECT_DEPTH:
        return None
    return (tuple(int(v) for v in cnt[d[0][2]][0]),
            tuple(int(v) for v in cnt[d[1][2]][0]))


def split_touching(cnt, min_area=None, depth=0):
    """Tach contour cua hai vat cung mau cham nhau thanh cac contour rieng.

    Nguyen ly: hop cua hai hinh loi tao ra dung hai diem lom sau tai cho chung
    giao nhau. Cat mask theo duong noi hai diem do se tra lai hai vat ban dau.
    De quy toi SPLIT_MAX_DEPTH lan de xu ly blob gom hon hai vat.

    Tra ve list contour theo toa do goc (chinh contour ban dau neu khong tach duoc).
    """
    min_area = cfg.MIN_AREA if min_area is None else min_area
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 1.0
    if depth >= cfg.SPLIT_MAX_DEPTH or solidity > cfg.SPLIT_SOLIDITY_MAX:
        return [cnt]

    pts = _deep_defects(cnt)
    if pts is None:
        return [cnt]

    # Ve blob vao ROI co dem vien, cat, roi tach thanh phan lien thong
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 2
    roi = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    off = (x - pad, y - pad)
    cv2.drawContours(roi, [cnt - off], -1, 255, cv2.FILLED)
    cv2.line(roi, (pts[0][0] - off[0], pts[0][1] - off[1]),
             (pts[1][0] - off[0], pts[1][1] - off[1]), 0, 3)

    n, labels = cv2.connectedComponents(roi)
    parts = []
    for i in range(1, n):
        m = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(m) < min_area:
            continue
        cc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cc:
            parts.append(c + off)

    if len(parts) < 2:
        return [cnt]   # cat khong ra hai manh dang ke -> giu nguyen
    out = []
    for pcnt in parts:
        out += split_touching(pcnt, min_area, depth + 1)
    return out
