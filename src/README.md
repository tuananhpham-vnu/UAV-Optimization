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

## Detection — 3 bước

1. **Mask bão hoà**: băng chuyền xám có `S` thấp, sản phẩm màu rực có `S` cao →
   một lần `inRange` trên kênh S tách nền sạch hơn và nhanh hơn MOG2 với camera cố định.
2. **Tách theo dải Hue**: hai sản phẩm khác màu chạm nhau sẽ bị gộp thành một
   contour nếu dùng chung một mask. Cắt mask theo Hue giúp chúng tách rời ngay từ
   bước mask (xem `#20`/`#21` trong video demo).
3. **Tách hình học** (`split_touching`): hai vật **cùng màu** chạm nhau thì bước 2
   bó tay. Hợp của hai hình lồi luôn tạo ra đúng hai điểm lõm sâu tại chỗ giao;
   cắt mask theo đường nối hai điểm đó trả lại hai vật ban đầu. Chỉ chạy trên
   contour có `solidity = A/A_hull < 0.93` nên khung hình bình thường không tốn
   thêm chi phí. Đệ quy tối đa `SPLIT_MAX_DEPTH` lần để xử lý blob gồm hơn hai vật.

## Classification — đặc trưng dùng

Không cần training. Cách làm: **chấm điểm khớp với từng primitive bằng IoU**
(intersection over union giữa mask của contour và mask của primitive khớp nhất).

| Primitive | Cách dựng |
|---|---|
| circle | `minEnclosingCircle` |
| square | `minAreaRect` |
| triangle | convex hull rút về đúng 3 đỉnh (xem cảnh báo bên dưới) |

Đo trên `conveyor_2d.mp4`:

| Contour thật là | IoU circle | IoU square | IoU triangle |
|---|---|---|---|
| circle | **0.945–0.992** | 0.79–0.81 | 0.02–0.37 |
| square | 0.63–0.70 | **0.913–0.995** | 0.48–0.51 |
| triangle | 0.45–0.52 | 0.52–0.57 | **0.826–0.918** |
| *blob 2 vật cùng màu chưa tách được* | 0.67 | 0.70 | 0.50 |

Lấy primitive điểm cao nhất; nếu điểm cao nhất `< SHAPE_FIT_MIN` (0.78) → gán
`unknown`, **không đếm**. Dùng IoU chứ không phải tỉ lệ diện tích vì IoU đối xứng:
nó phạt cả khi primitive to hơn vật lẫn khi nhỏ hơn vật, nên blob hỏng không thể
ăn điểm cao ở bất kỳ primitive nào.

Điểm quan trọng: cách cũ so `circularity`/`extent` với bảng tham chiếu rồi *luôn
luôn* chọn nhãn gần nhất, nên blob gộp từ square + triangle đỏ bị gán thành
`red_circle` dù `circularity` chỉ 0.43. Chấm điểm khớp phát hiện được "không giống
cái nào" thay vì bịa ra nhãn.

> **Cảnh báo — không dùng `cv2.minEnclosingTriangle`.**
> Hàm này trả về kết quả sai trên một số point set. Trường hợp gặp thật trong
> video: tam giác diện tích 5811 px được báo là bị bao bởi tam giác 8694 px, với
> một đỉnh nằm ở `x=166` trong khi convex hull chỉ kéo tới `x=111`. Hệ quả là
> tam giác thật bị chấm 0.67 và rơi vào `unknown` y như blob hỏng. `_triangle_of()`
> dựng tam giác bằng cách nhị phân tìm epsilon cho `approxPolyDP` trên convex hull
> đến khi còn đúng 3 đỉnh (có nhánh dự phòng chọn 3 đỉnh hull cho diện tích lớn nhất).

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
- 40 sản phẩm vượt vạch, 14 tổ hợp nhãn `màu_hình`
- 244 lần `split_touching` cắt được blob cùng màu; 0 blob bị từ chối lúc qua vạch
- 14/8026 detection (0.17%) còn `unknown` ở một số frame trung gian — chồng lấn
  quá sâu, không frame nào trong số đó rơi đúng lúc qua vạch đếm
- Không có ID nào đếm trùng (mỗi ID chỉ cập nhật bộ đếm 1 lần)

## Giới hạn đã biết (đúng phạm vi MVP1)

- `split_touching` cắt theo **một** đường thẳng nên xử lý được hai vật lồi chạm
  nhau; ba vật chồng chéo phức tạp vẫn có thể còn dính sau khi đệ quy hết độ sâu.
- Vật bị che quá sâu (chỉ còn thấy một mẩu) vẫn ra `unknown` — đúng thiết kế,
  vì lúc đó không đủ thông tin hình học để kết luận.
- Bounding box của vật **bị che** là phần nhìn thấy được sau khi cắt, hơi nhỏ hơn
  vật thật. Nhãn và số đếm đúng, nhưng đừng dùng `area_px` của nó để đo kích thước.
- Ngưỡng màu/diện tích tune theo video giả lập này; quay video thật cần tune lại
  `SAT_MIN`, `VAL_MIN` và các ngưỡng size trong `config.py`.
- Chưa đo được Detection/Classification accuracy vì video chưa có ground truth —
  theo kế hoạch cần bảng ghi tay (thứ tự thả, class thật) để đối chiếu với CSV.
