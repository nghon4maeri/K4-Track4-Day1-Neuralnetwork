# Báo cáo Lab Day 1 — 2A202602788

## 1. Thiết lập

- Môi trường: Máy local, NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.x
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203. Validation: 20% của train (371 847 train / 92 962 val).
- Model: `M-base` (54→256→128→7, 47 879 tham số), `M-wide`, `M-deep`.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0,4876.
- Các chủ đề đã thử: [x] loss [x] optimizer [x] hyper-parameter [x] dropout [x] clipping [x] init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | ~1.946 |
| Quá khớp 20 mẫu: loss cuối | ~0.0001 |
| Mọi tham số có gradient khác 0 | [x] có |
| Baseline, số seed đã chạy | 2 |
| Baseline: val macro-F1 (TB ± σ) | 0.8764 ± 0.0006 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0.0012 (val macro-F1). 

## 3. Kết quả theo chủ đề

### 3.2 Bộ tối ưu hoá (Adam vs AdamW vs SGD)
- **Dự đoán**: Adam sẽ hội tụ nhanh hơn SGD. AdamW sẽ giúp mô hình tổng quát hoá (generalize) tốt hơn Adam nhờ cơ chế Weight Decay độc lập.
- **Kết quả (`opt-adam-lr1e-3` & `opt-adamw-wd0.01`)**: 
  - Adam (lr=1e-3): Val Macro-F1 = 0.8709
  - AdamW (lr=1e-3, wd=0.01): Val Macro-F1 = 0.8727
  - SGD Baseline (lr=0.1): Val Macro-F1 ~ 0.8764
- **Giải thích**: Sau 40 epochs, SGD với Momentum vẫn cho kết quả tiệm cận hoặc nhỉnh hơn Adam một chút về độ tổng quát, dù Adam hội tụ nhanh hơn ở những epoch đầu. AdamW chứng minh được ưu điểm chống quá khớp (overfitting) tốt hơn Adam thuần túy (F1 cao hơn 0.0018, vượt qua ngưỡng nhiễu 0.0012).

### 3.3 Kiến trúc mô hình (Hyper-parameter)
- **Yếu tố thay đổi**: Tăng số lượng nơ-ron (M-wide: 512, 256) và tăng số tầng ẩn (M-deep: 256, 128, 64).
- **Kết quả**: 
  - M-wide (`arch-m-wide`): Val Macro-F1 vọt lên mức **0.9040**.
  - M-deep (`arch-m-deep`): Val Macro-F1 đạt **0.8861**.
- **Giải thích**: Số lượng dữ liệu dồi dào (~370k mẫu huấn luyện) cho phép các mô hình có dung lượng biểu diễn (capacity) lớn hơn học được các ranh giới phân loại phức tạp. `M-wide` với 161,287 tham số đã tận dụng tốt lượng dữ liệu này để tăng độ chính xác vượt bậc.

### 3.4 Dropout
- **Dự đoán**: Mô hình chưa bị quá khớp (overfit) nghiêm trọng, nên dùng Dropout có thể làm mô hình khó học hơn.
- **Kết quả (`reg-dropout-0.3`)**: Val Macro-F1 rớt xuống **0.8364**.
- **Giải thích**: Do mạng `M-base` có số lượng tham số khá khiêm tốn so với lượng dữ liệu khủng (47k tham số vs 370k dữ liệu), hiện tượng Overfit hầu như chưa xảy ra (Train loss và Val loss vẫn bám sát nhau). Việc tắt đi 30% nơ-ron (Dropout 0.3) làm giảm khả năng học của mạng, dẫn đến hiện tượng underfitting (chưa khớp).

### 3.5 Gradient Clipping
- **Kết quả (`reg-clip-1.0`)**: Val Macro-F1 đạt 0.8755 (tương đồng với baseline).
- **Giải thích**: Trong điều kiện huấn luyện bình thường với hàm CrossEntropy và `M-base`, gradient norm hiếm khi vượt quá 1.0 (như biểu đồ đo lường), do đó clipping gần như không được kích hoạt, kết quả không khác biệt so với không clip.

## 4. Đánh giá cuối trên tập eval

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Cấu hình cuối cùng (M-wide + Adam) | 1 | 0.9040 | **0.9018** | 0.9337 |

- Cấu hình cuối cùng được chọn là **M-wide (512, 256) kết hợp Optimizer Adam**. Cấu hình này tỏ ra vượt trội hoàn toàn trên tập Validation so với mọi thiết lập khác (cải thiện hơn 0.027 so với Baseline, khác biệt này hoàn toàn có ý nghĩa và vượt xa độ nhiễu 0.0012).
- Điểm đánh giá thực tế trên tập Eval ẩn (0.9018) gần như trùng khớp hoàn hảo với tập Val (0.9040), chứng tỏ chiến lược chia (stratify) và chuẩn hoá của bài Lab cực kỳ vững chắc.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 42368 | 0.9448 | 0.9136 | 0.9289 |
| 1 | 56661 | 0.9305 | 0.9569 | 0.9435 |
| 2 |  7151 | 0.9261 | 0.9466 | 0.9362 |
| 3 |   549 | 0.8452 | 0.8652 | 0.8551 |
| 4 |  1899 | 0.8914 | 0.7778 | 0.8307 |
| 5 |  3473 | 0.8890 | 0.8624 | 0.8755 |
| 6 |  4102 | 0.9466 | 0.9386 | 0.9426 |

- Lớp khó nhất là lớp 4 (F1 = 0.8307). Dựa vào ma trận nhầm lẫn (Confusion Matrix), nó bị nhầm thành lớp 1 (360 mẫu) và lớp 0 (30 mẫu).
- Lý giải: Lớp 4 là lớp vô cùng thiểu số. Dù kiến trúc `M-wide` đã cải thiện F1 của lớp này (từ ~0.75 ở Baseline lên 0.83), mô hình vẫn còn chút định kiến (bias) dự đoán thiên về các lớp phổ biến nhất (0 và 1).
- Cải thiện: Có thể thử các phương pháp Over-sampling, Under-sampling hoặc thêm Class Weight trong hàm Loss.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng?**: Adam hội tụ nhanh ở pha đầu, nhưng SGD+Momentum vươn lên đuổi kịp và nhỉnh hơn một chút ở pha sau. Tuy nhiên khi thay đổi kiến trúc mô hình (M-wide), Adam lại giúp các mô hình phức tạp hội tụ một cách rất hiệu quả.
2. **Dropout có giúp không?**: Với mô hình `M-base` và dữ liệu CoverType, Dropout làm giảm độ chính xác vì mô hình chưa gặp tình trạng Overfitting. Dropout chỉ thực sự hữu dụng nếu Training Loss xuống rất thấp nhưng Validation Loss có dấu hiệu tăng vọt trở lại.
3. **Gradient clipping giải quyết vấn đề gì?**: Giải quyết việc Loss bùng nổ (NaN/inf) do gradient vượt quá tầm kiểm soát (Exploding Gradient), đặc biệt hữu ích khi thử Learning Rate cực cao hoặc trong mạng RNN.
5. **Vì sao khởi tạo toàn số 0 hỏng?**: Khởi tạo trọng số bằng 0 gây ra hiện tượng đối xứng (Symmetry Breaking failure); các nơ-ron cùng một lớp sẽ luôn nhận chung một giá trị cập nhật, khiến toàn bộ tầng ẩn đó hành xử y hệt như một nơ-ron duy nhất.

## 6. Hạn chế và điều bất ngờ
- **Bất ngờ**: Mạng `M-wide` cho độ cải thiện sức mạnh kinh ngạc, chứng minh được dung lượng mô hình (model capacity) ban đầu bị kìm hãm so với lượng dữ liệu phong phú của Forest CoverType.
- **Hạn chế**: Số lượng Epochs dừng ở mức 40. Nếu tăng lên 100 epochs kèm thêm Learning Rate Scheduler (giảm LR dần dần), điểm số có thể phá vỡ mốc 0.92 Macro-F1.

## 7. Phụ lục
- `experiments.xlsx`
- `predictions_eval.csv`
- `eval_result.json`
- Thư mục `figures/`
- Thư mục `code/`
