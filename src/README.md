# MVP1 — Nhóm 2: Detection & Classification

Pipeline OpenCV classical chạy trên `conveyor_2d.mp4` (băng chuyền 2D giả lập,
1280x480 @ 30 FPS, 600 frame).

## Chạy

```bash
python -m pip install -r requirements.txt
python src/demo.py                 # xuất outputs/demo_annotated.mp4 + outputs/log.csv
python src/demo.py --show          # xem trực tiếp (phím q để thoát)
python src/demo.py --mask          # xuất thêm video mask tách nền để soi lỗi
python src/demo.py --video other.mp4
```

## Kiến trúc

| File | Vai trò |
|---|---|
| `config.py` | Toàn bộ ngưỡng — tune ở đây, không sửa code logic |
| `detector.py` | **Detection**: tiền xử lý → tách nền → contour → bbox + centroid |
| `classifier.py` | **Classification**: trích đặc trưng → nhãn hình dạng / màu / kích thước |
| `tracker.py` | Centroid tracker tối thiểu: ID + chặn đếm trùng (bàn giao cho nhánh tracking) |
| `demo.py` | Ghép luồng, vẽ overlay, ghi CSV, đo FPS |

## Detection — 2 bước

1. **Mask bão hoà**: băng chuyền xám có `S` thấp, sản phẩm màu rực có `S` cao →
   một lần `inRange` trên kênh S tách nền sạch hơn và nhanh hơn MOG2 với camera cố định.
2. **Tách theo dải Hue**: hai sản phẩm khác màu chạm nhau sẽ bị gộp thành một
   contour nếu dùng chung một mask. Cắt mask theo Hue giúp chúng tách rời ngay từ
   bước mask (xem `#20`/`#21` trong video demo).

## Classification — đặc trưng dùng

Không cần training. Cách làm: **chấm điểm khớp với từng primitive** —
`điểm = diện tích contour / diện tích primitive bao quanh nó`, khớp hoàn hảo = 1.0.

| Primitive | Hàm OpenCV |
|---|---|
| circle | `minEnclosingCircle` |
| square | `minAreaRect` |
| triangle | `minEnclosingTriangle` |

Đo trên `conveyor_2d.mp4`:

| Contour thật là | fit_circle | fit_square | fit_triangle |
|---|---|---|---|
| circle | **0.93–0.96** | 0.79–0.81 | 0.60–0.61 |
| square | 0.61–0.68 | **0.92–1.00** | 0.49–0.53 |
| triangle | 0.43–0.47 | 0.52–0.56 | **0.91–0.98** |
| *blob 2 vật cùng màu chạm nhau* | 0.42 | 0.68 | 0.54 |

Lấy primitive điểm cao nhất; nếu điểm cao nhất `< SHAPE_FIT_MIN` (0.85) → gán
`unknown`. Đây là điểm quan trọng: cách cũ so `circularity`/`extent` với bảng
tham chiếu rồi *luôn luôn* chọn nhãn gần nhất, nên một blob gộp từ square + triangle
đỏ bị gán thành `red_circle` dù `circularity` chỉ 0.43. Chấm điểm khớp phát hiện
được "không giống cái nào" thay vì bịa ra nhãn.

- **Màu**: lấy từ dải Hue ở bước tách mask, kiểm chứng bằng Hue trung vị bên trong
  contour (trung vị chứ không phải trung bình — tránh lệch do viền khử răng cưa).
- **Kích thước**: ngưỡng diện tích → `S` / `M` / `L`.
- **Bỏ phiếu qua frame**: nhãn của mỗi ID là nhãn được bầu nhiều nhất qua các frame
  nó xuất hiện → ổn định hơn phán đoán trên một frame đơn lẻ.
- **Vật bị cắt ở mép khung** bị đánh dấu `truncated`, gán `unknown` và không tính
  vào bộ đếm — đặc trưng hình học của vật cụt không đáng tin.

## Đầu ra

`outputs/log.csv` — mỗi dòng là một sản phẩm vượt vạch đếm:

```
timestamp_s,frame,id,shape,color,size,label,area_px,cx,cy,shape_conf,count_total
```

`outputs/log_rejected.csv` — chỉ sinh ra khi có blob bị từ chối. Ghi lại điểm
khớp của cả 3 primitive để soi nguyên nhân, **không** tính vào bộ đếm:

```
timestamp_s,frame,id,color,area_px,cx,cy,best_fit,fit_circle,fit_square,fit_triangle
15.733,472,29,red,12602,645,251,0.685,0.423,0.685,0.535
```

## Kết quả trên conveyor_2d.mp4

- 600 frame, **~98 FPS** phần thuật toán (KPI ≥ 30 FPS)
- 38 sản phẩm vượt vạch, 14 tổ hợp nhãn `màu_hình`
- 1 blob bị từ chối (square + triangle đỏ chạm nhau, `best_fit = 0.685`)
- Không có ID nào đếm trùng (mỗi ID chỉ cập nhật bộ đếm 1 lần)

## Giới hạn đã biết (đúng phạm vi MVP1)

- Hai vật **cùng màu** chồng lên nhau vẫn bị gộp thành một contour. Hệ thống
  **phát hiện được** và gán `unknown` thay vì gán nhãn sai, nhưng chưa tách được
  chúng ra — cả hai đều không được đếm. Hướng xử lý tiếp: watershed trên distance
  transform, hoặc template matching để tách blob thành 2 primitive.
- Ngưỡng màu/diện tích tune theo video giả lập này; quay video thật cần tune lại
  `SAT_MIN`, `VAL_MIN` và các ngưỡng size trong `config.py`.
- Chưa đo được Detection/Classification accuracy vì video chưa có ground truth —
  theo kế hoạch cần bảng ghi tay (thứ tự thả, class thật) để đối chiếu với CSV.
