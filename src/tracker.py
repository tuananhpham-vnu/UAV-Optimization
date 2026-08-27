"""Centroid tracker toi thieu: gan ID on dinh + chan dem trung.

Muc dich trong pham vi Nhom 2: co ID de (1) bo phieu nhan qua nhieu frame,
giup phan loai on dinh hon, va (2) xuat CSV theo dung format ban giao.
Phan tracking day du (quy dao, occlusion) thuoc nhanh con lai.
"""

from collections import Counter

import config as cfg


class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}   # id -> dict(centroid, missing, votes, counted, prev_x)

    def update(self, detections):
        """Gan ID cho detections cua frame hien tai (sua truc tiep vao dict)."""
        used = set()
        for oid, obj in self.objects.items():
            obj["matched"] = False

        for det in detections:
            cx, cy = det["centroid"]
            best_id, best_d = None, cfg.MATCH_MAX_DIST ** 2
            for oid, obj in self.objects.items():
                if oid in used:
                    continue
                ox, oy = obj["centroid"]
                d = (cx - ox) ** 2 + (cy - oy) ** 2   # so sanh binh phuong, bo sqrt
                if d < best_d:
                    best_id, best_d = oid, d

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.objects[best_id] = {
                    "centroid": (cx, cy),
                    "missing": 0,
                    "votes": Counter(),
                    "counted": False,
                    "prev_x": cx,
                }

            obj = self.objects[best_id]
            obj["prev_x"] = obj["centroid"][0]
            obj["centroid"] = (cx, cy)
            obj["missing"] = 0
            obj["matched"] = True
            used.add(best_id)

            # Bo phieu nhan qua nhieu frame -> nhan on dinh hon 1 frame don le
            if not det["truncated"]:
                obj["votes"][(det["shape"], det["color"], det["size"])] += 1
            if obj["votes"]:
                shape, color, size = obj["votes"].most_common(1)[0][0]
                det["shape"], det["color"], det["size"] = shape, color, size
                det["label"] = f"{color}_{shape}"

            det["id"] = best_id

        # Xoa cac ID mat dau qua lau
        for oid in list(self.objects):
            if not self.objects[oid].get("matched", False):
                self.objects[oid]["missing"] += 1
                if self.objects[oid]["missing"] > cfg.MAX_MISSING:
                    del self.objects[oid]
        return detections

    def check_line_crossing(self, det, line_x):
        """True dung 1 lan duy nhat khi tam san pham vuot vach dem."""
        obj = self.objects.get(det.get("id"))
        if obj is None or obj["counted"] or det["truncated"]:
            return False
        prev_x, cur_x = obj["prev_x"], det["centroid"][0]
        if (prev_x < line_x <= cur_x) or (prev_x > line_x >= cur_x):
            obj["counted"] = True
            return True
        return False
