# Hướng dẫn chạy — MVP1 Nhóm 2 (Detection & Classification)

## 1. Yêu cầu

- Python 3.11 (đã test trên 3.11.9, Windows)
- `data/conveyor_2d.mp4` có trong repo

## 2. Cài đặt

Lần đầu, tạo môi trường ảo và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell / CMD)
# source .venv/bin/activate      # macOS / Linux
python -m pip install -r requirements.txt
```

`requirements.txt` chỉ có `opencv-python` và `numpy` — pipeline là OpenCV
classical, **không cần GPU, không cần model weight, không cần train**.

## 3. Chạy demo

```bash
python src/demo.py
```

Chạy ~10 giây, kết thúc sẽ in ra bảng thống kê và ghi 2 file vào `outputs/`.

### Các tuỳ chọn

| Lệnh | Tác dụng |
|---|---|
| `python src/demo.py` | Chạy mặc định, xuất video + CSV |
| `python src/demo.py --show` | Mở cửa sổ xem trực tiếp (bấm **q** để thoát) |
| `python src/demo.py --mask` | Xuất thêm video mask tách nền để soi lỗi detection |
| `python src/demo.py --video data/khac.mp4` | Chạy trên video khác |
| `python src/demo.py --out a.mp4 --csv b.csv` | Đổi đường dẫn đầu ra |

Các tuỳ chọn ghép được với nhau, ví dụ `python src/demo.py --show --mask`.

## 4. Kết quả mong đợi

Terminal in ra:

```
Frames xu ly      : 600
FPS xu ly (thuat toan): 77.8
FPS ke ca I/O ghi video: 53.0
Tong san pham dem duoc : 39
         blue_circle: 5
         blue_square: 1
       ...
```

Sinh ra trong `outputs/`:

| File | Nội dung |
|---|---|
| `demo_annotated.mp4` | Video có bbox, nhãn `#ID màu_hình kích-thước`, vạch đếm, bảng FPS/thống kê |
| `log.csv` | Mỗi dòng = 1 sản phẩm vượt vạch đếm |
| `log_rejected.csv` | Chỉ sinh ra khi có blob bị từ chối (xem mục 4.1) |
| `demo_annotated_mask.mp4` | Chỉ có khi chạy `--mask` — mask nhị phân sau tách nền |

Format `log.csv`:

```
timestamp_s,frame,id,shape,color,size,label,area_px,cx,cy,shape_conf,count_total
3.9,117,1,triangle,red,S,red_triangle,1801,641,249,0.929,1
4.3,129,2,square,red,M,red_square,3652,641,379,0.95,2
```

### 4.1. Dòng "Blob bi tu choi"

Nếu terminal in thêm:

```
Blob bi tu choi (khong khop hinh nao): 1
  frame  472  #29  red      best_fit=0.685
```

nghĩa là có contour không khớp hình tròn/vuông/tam giác nào đủ tốt — gần như luôn
là **hai sản phẩm cùng màu chạm nhau bị gộp làm một**. Hệ thống gán `unknown`,
vẽ khung xám, và **không đếm** — cố ý như vậy, vì gán đại một nhãn sẽ làm sai cả
Classification lẫn Counting. Chi tiết điểm khớp nằm trong `outputs/log_rejected.csv`.

Nếu số blob bị từ chối nhiều bất thường trên video tự quay, xem mục 5 —
thường là do `SAT_MIN` quá thấp làm nền dính vào contour, chứ không phải do vật chạm nhau.

Con số FPS sẽ khác nhau tuỳ máy — KPI cần đạt là **≥ 30 FPS** ở dòng
*"FPS xu ly (thuat toan)"* (dòng còn lại có tính cả thời gian ghi file video,
không phải chi phí của thuật toán).

## 5. Chạy trên video tự quay

Video thật có nền và ánh sáng khác video giả lập, nên phải tune lại ngưỡng.
**Sửa `src/config.py`, không sửa code logic.** Quy trình:

1. Chạy `python src/demo.py --video data/video_cua_ban.mp4 --mask`
2. Mở `outputs/demo_annotated_mask.mp4` xem mask tách nền có sạch không:
   - Nền lọt vào mask (nhiều đốm trắng rác) → **tăng** `SAT_MIN`
   - Sản phẩm bị thủng lỗ / mất mảng → **giảm** `SAT_MIN` hoặc `VAL_MIN`
   - Còn nhiễu hạt li ti → tăng `MORPH_KERNEL` hoặc `MIN_AREA`
3. Bbox nhảy lung tung giữa các frame → tăng `MATCH_MAX_DIST`
4. Hình thật bị gán `unknown` (khung xám) → **giảm** `SHAPE_FIT_MIN`;
   ngược lại vật chạm nhau vẫn lọt nhãn sai → **tăng** `SHAPE_FIT_MIN`
   (đối chiếu cột `fit_*` trong `log_rejected.csv`)
5. Nhãn `S`/`M`/`L` lệch → chỉnh `SIZE_SMALL_MAX`, `SIZE_MEDIUM_MAX`
   (xem cột `area_px` trong CSV để biết diện tích thật của từng loại)
6. Màu sản phẩm không nằm trong 5 dải mặc định → sửa `COLOR_RANGES`
   (Hue trong OpenCV là 0..179, **không phải** 0..359)

Vạch đếm mặc định ở giữa khung (`COUNT_LINE_RATIO = 0.5`).

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Khong mo duoc video: ...` | Sai đường dẫn, hoặc chạy `demo.py` từ thư mục khác — phải chạy từ **thư mục gốc** của repo |
| `ModuleNotFoundError: cv2` | Chưa activate `.venv`, hoặc chưa `pip install -r requirements.txt` |
| `ModuleNotFoundError: config` | Đang chạy `cd src && python demo.py` — hãy chạy `python src/demo.py` từ gốc repo |
| Cửa sổ `--show` không hiện | Chạy trong môi trường không có GUI (SSH/WSL không X11) — bỏ `--show`, xem file video xuất ra |
| Đếm thiếu/thừa so với thực tế | Xem mục 5, tune `SAT_MIN` và `MIN_AREA` trước |
| Vật hiện khung xám, nhãn `unknown` | Đúng thiết kế khi hai vật cùng màu chạm nhau — xem mục 4.1 |

## 7. Web demo (Streamlit)

Ngoài CLI, có sẵn app web để demo trước lớp — chọn video, kéo slider tune ngưỡng,
xem pipeline chạy trực tiếp, tải về CSV và video đã gán nhãn.

### Chạy local

```bash
python -m pip install -r requirements-web.txt
streamlit run app.py
```

Mở http://localhost:8501.

### Deploy lên Render

Repo đã có sẵn `render.yaml` (Blueprint), không cần cấu hình tay:

1. Push repo lên GitHub — **nhớ commit cả `data/conveyor_2d.mp4`**, không có file này
   thì app không có video mẫu.
2. Vào Render Dashboard → **New** → **Blueprint** → chọn repo.
3. Render đọc `render.yaml` và tự dựng service. Lần build đầu mất vài phút.

Một số điểm trong `render.yaml` cần biết khi sửa:

| Mục | Lý do |
|---|---|
| `buildCommand` dùng `requirements-web.txt` | File này cài `opencv-python-headless`. Bản đầy đủ `opencv-python` cần thư viện GUI của hệ điều hành, server Render không có → lỗi `ImportError: libGL.so.1` |
| `--server.address 0.0.0.0` và `--server.port $PORT` | Render cấp cổng động qua biến `$PORT`, bind vào `localhost` sẽ bị báo "no open ports detected" |
| `healthCheckPath: /_stcore/health` | Endpoint health có sẵn của Streamlit |
| `region: singapore` | Gần Việt Nam nhất; đổi được sang `oregon`, `frankfurt`, `ohio`, `virginia` |
| `plan: free` | Đổi sang `starter` nếu cần máy mạnh hơn |

### Ba giới hạn của gói free cần lưu ý khi demo

- **CPU yếu** (0.1 CPU): FPS trên web sẽ thấp hơn nhiều so với máy local và có thể
  tụt dưới KPI 30 — app sẽ tự hiện cảnh báo. **Số liệu KPI nghiệm thu phải đo bằng
  CLI trên máy thật**, không lấy con số hiển thị trên web.
- **Ngủ sau 15 phút không dùng**: lần truy cập kế tiếp mất ~30-60 giây để dậy.
  Trước khi demo, mở link trước vài phút cho service khởi động.
- **Ổ đĩa tạm**: file trong `outputs/` trên server sẽ mất khi service restart.
  App vì vậy trả kết quả qua nút tải về chứ không lưu trên server.

## 8. Tài liệu liên quan

- `src/README.md` — giải thích thuật toán: đặc trưng phân loại, cách tách vật chạm nhau, giới hạn đã biết
- `docs/KeHoach_TongQuan_MVP1.pdf` — kế hoạch tổng quan, KPI nghiệm thu
- `docs/Khao Sat Dataset MVP1.pdf` — khảo sát dataset, class schema
