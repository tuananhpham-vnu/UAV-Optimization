"""Core Detection: tien xu ly -> tach nen -> tim contour -> bounding box.

Chien luoc 2 buoc (nhanh + tach duoc vat cham nhau):
  1. Mask bao hoa: bang chuyen xam (S thap) bi loai, san pham mau ruc (S cao) giu lai.
     Chi 1 lan inRange + 1 lan morphology tren toan khung -> re.
  2. Tach mask do theo tung dai Hue: hai san pham khac mau cham nhau se thanh
     hai contour rieng thay vi bi gop lam mot.

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
            for cnt in contours:
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
