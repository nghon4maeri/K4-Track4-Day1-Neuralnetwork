# Báo cáo Lab Day 1 — 2A202602788

## 1. Thiết lập

- Môi trường: Máy local, CPU/GPU, PyTorch 2.x
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203. Validation: 20% của train (371 847 train / 92 962 val).
- Model: `M-base` (54→256→128→7, 47 879 tham số).
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val ≈ 0,4876.
- Các chủ đề đã thử: optimizer

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | ~1,946 |
| Quá khớp 20 mẫu: loss cuối | ~0.0001 |
| Mọi tham số có gradient khác 0 | [x] có |
| Baseline, số seed đã chạy | 2 |

*(Chưa ghi nhận đủ số seed để tính độ nhiễu chính xác tuyệt đối, nhưng độ biến thiên của SGD dao động khoảng ±0.005 Macro-F1).*

## 3. Kết quả theo chủ đề

### 3.2 Bộ tối ưu hoá
- **Dự đoán**: Adam sẽ hội tụ nhanh hơn SGD vì nó có khả năng tùy chỉnh learning rate riêng cho từng tham số (adaptive learning rate) và vượt qua được các local minima hiệu quả hơn.
- **Kết quả (`opt-adam-lr1e-3`)**: Adam (lr=1e-3) đạt Val Macro-F1 ~ 0.85, cao hơn SGD Baseline (lr=0.1).
- **Giải thích**: SGD cập nhật gradient đồng đều cho mọi tham số, trong khi Adam sử dụng moment bậc 1 và moment bậc 2 để chia scale cho gradient. Điều này giúp Adam đi nhanh ở những chiều có độ dốc ổn định và chậm lại ở những chiều gradient biến thiên mạnh.

## 4. Đánh giá cuối trên tập eval

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Cấu hình cuối cùng (Adam) | 1 | ~0.8500 | **0.8507** | 0.9022 |

- Cấu hình cuối cùng được chọn là **Adam Optimizer (lr=1e-3)** vì nó cho ra loss hội tụ tốt nhất trên tập Validation.
- Val và Eval cực kỳ sát nhau (0.8500 vs 0.8507), chứng tỏ mô hình có tính tổng quát hoá cao (generalization tốt), tập Validation được phân tầng (stratified split) rất đồng đều so với tập Test ẩn.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 42368 | 0.9131 | 0.8845 | 0.8986 |
| 1 | 56661 | 0.9005 | 0.9361 | 0.9179 |
| 2 |  7151 | 0.8937 | 0.8755 | 0.8845 |
| 3 |   549 | 0.8508 | 0.7377 | 0.7902 |
| 4 |  1899 | 0.8463 | 0.6872 | 0.7585 |
| 5 |  3473 | 0.8246 | 0.7596 | 0.7908 |
| 6 |  4102 | 0.9217 | 0.9066 | 0.9141 |

- **Lớp khó đoán nhất là lớp 4** (F1 = 0.7585) và **lớp 3** (F1 = 0.7902). 
- Dựa vào ma trận nhầm lẫn (Confusion Matrix), **Lớp 4 hay bị nhầm với Lớp 1 (510 mẫu) và Lớp 0 (61 mẫu)**. 
- **Lý giải**: Lớp 3 và Lớp 4 là những lớp hiếm (minority classes) với số lượng mẫu (support) rất ít (chỉ 549 và 1899 mẫu so với >50k mẫu của lớp 1). Do mất cân bằng dữ liệu nghiêm trọng, mô hình có xu hướng thiên vị (bias) và dự đoán nhầm lớp thiểu số thành lớp đa số (như lớp 1).
- **Cải thiện đề xuất**: Thử dùng Focal Loss hoặc thêm kỹ thuật class weight (trọng số lớp) vào hàm CrossEntropy để phạt nặng hơn khi mô hình dự đoán sai các lớp hiếm.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng?**: Adam cho thấy độ ổn định cao và hội tụ nhanh hơn SGD. 
6. **Mạng có loss không giảm sau 2000 bước. 3 phép kiểm tra đầu tiên là gì?**
   - *Kiểm tra Loss bước 0*: Xem thử loss ban đầu có xấp xỉ `-ln(1/C)` không. Nếu lớn hơn nhiều, hàm khởi tạo trọng số bị sai (làm logits đầu ra quá lớn).
   - *Kiểm tra Overfit trên 1 batch nhỏ (20 mẫu)*: Tắt hết dropout, overfit thử xem loss có về 0 không. Nếu không, có thể code bị bug (nhãn bị lệch, softmax 2 lần, hoặc quên `zero_grad`).
   - *Kiểm tra norm của gradient*: In ra `grad_norm` của các tham số. Nếu norm = 0 hoặc None, chứng tỏ gradient không chảy được ngược về các lớp dưới (ví dụ do ReLU bị chết hoặc kiến trúc bị ngắt kết nối).

## 6. Hạn chế và điều bất ngờ
- **Hạn chế**: Số lượng seed đo lường còn ít, cũng như số lượng epoch (20) là chưa đủ để các thuật toán hội tụ hoàn toàn.
- **Dự định tiếp theo**: Thay đổi cấu trúc thành mạng sâu hơn (`M-deep`) kết hợp thêm `Dropout` để xem có vượt mức F1 0.90 hay không.

## 7. Phụ lục
- `experiments.xlsx`
- `predictions_eval.csv`
- `eval_result.json`
- Thư mục `figures/`
- Thư mục `code/`
