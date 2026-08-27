"""Tham so cau hinh cho pipeline detection + classification (MVP1 - Nhom 2)."""

# ---- Nguon video ----
VIDEO_PATH = "data/conveyor_2d.mp4"
OUT_VIDEO = "outputs/demo_annotated.mp4"
OUT_CSV = "outputs/log.csv"

# ---- Tach nen (segmentation) ----
# Bang chuyen co mau xam (do bao hoa thap), san pham mau ruc (do bao hoa cao)
# -> nguong tren kenh S cua HSV tach nen rat sach va nhanh hon MOG2.
SAT_MIN = 90          # nguong bao hoa toi thieu de coi la san pham
VAL_MIN = 60          # loai bo vung qua toi
MORPH_KERNEL = 5      # kich thuoc kernel open/close khu nhieu
MIN_AREA = 400        # dien tich contour toi thieu (px) - loai nhieu vun
MAX_AREA = 200_000    # dien tich toi da - loai vung nen loi

# ---- Tach vat cung mau cham nhau (convexity-defect split) ----
# Hinh loi don le co solidity = A/A_hull ~ 1.0; blob do hai vat cham nhau tut
# xuong ~0.87 vi phan lom o cho giao nhau. Chi blob duoi nguong moi bi thu tach
# -> khong ton chi phi cho khung hinh binh thuong.
SPLIT_SOLIDITY_MAX = 0.93
SPLIT_MIN_DEFECT_DEPTH = 8.0   # px - do sau toi thieu de coi la cho giao that
SPLIT_MAX_DEPTH = 2            # so lan tach de quy -> toi da 4 vat trong 1 blob

# ---- Phan loai hinh dang ----
# Diem khop = IoU giua contour va primitive khop nhat voi no.
# Do tren conveyor_2d.mp4: hinh that dat 0.86-0.99 voi primitive dung cua no,
# primitive sai chi dat <= 0.81, blob do nhieu vat cham nhau khong qua 0.71.
# Nguong 0.78 nam giua hai vung, co bien an toan ca hai phia.
SHAPE_FIT_MIN = 0.78

# ---- Phan loai kich thuoc (nguong dien tich, px) ----
SIZE_SMALL_MAX = 3000
SIZE_MEDIUM_MAX = 9000

# ---- Phan loai mau (khoang Hue trong OpenCV: 0..179) ----
COLOR_RANGES = [
    ("red",     [(0, 10), (170, 180)]),
    ("yellow",  [(20, 38)]),
    ("green",   [(40, 85)]),
    ("blue",    [(86, 130)]),
    ("magenta", [(131, 169)]),
]

# ---- Tracking / dem (toi thieu, de xuat CSV) ----
COUNT_LINE_RATIO = 0.5   # vach dem o giua khung hinh (theo truc X)
MATCH_MAX_DIST = 60      # khoang cach toi da de gan lai ID giua 2 frame
MAX_MISSING = 8          # so frame mat dau truoc khi xoa ID

# ---- Hien thi ----
DRAW_COLORS = {
    "circle":   (0, 255, 255),
    "square":   (0, 200, 0),
    "triangle": (255, 120, 0),
    "unknown":  (128, 128, 128),
}
