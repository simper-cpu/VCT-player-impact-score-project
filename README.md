# VCT Player Performance Forecasting

Pipeline dự đoán trước từng map cho ba target độc lập: `rating2_all`,
`acs_all` và `kda_all`. Pipeline dùng lịch sử player/team, Elo trước map,
split theo thời gian và không dùng deep learning.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

## Chạy từ đầu đến cuối

### 1. Chuẩn hóa dataset

```powershell
python src/preprocessing/build_player_match_dataset.py
```

Script loại duplicate theo `(player_id, match_id, map)`, loại duplicate map
key, parse target/numeric fields và tạo
`data/processed/player_match_dataset_cleaned.csv`. Dataset gốc không bị ghi
đè. Các aggregate tính trên toàn bộ dataset như `team_avg_rating` bị loại
khỏi artifact sạch vì gây leakage. Elo được tính trước map và chỉ cập nhật sau
khi map kết thúc.

### 2. Train, evaluate và export

```powershell
python -m src.models.train_models
```

Lệnh này tạo `data/processed/player_match_features.csv`, chia whole-match theo
thời gian 70%/15%/15%, so sánh baseline/Random Forest/XGBoost và CatBoost nếu
đã cài, thử nhiều cấu hình XGBoost, thử raw KDA/`log1p(KDA)`/Huber/quantile,
chọn shortlist bằng ba expanding rolling time windows, refit train +
validation và đánh giá test.

Model bundle chứa preprocessing, encoder, imputer, model và metadata:

```text
models/vct_rating_pipeline.joblib
models/vct_acs_pipeline.joblib
models/vct_kda_pipeline.joblib
```

Reports nằm trong `reports/modeling/`. Đặc biệt:

- `metrics_before_improvements.csv`: mốc trước khi thêm feature.
- `rolling_validation_metrics.csv`: lỗi theo từng giai đoạn thời gian.
- `rolling_selection.csv`: candidate được so sánh với global mean theo từng
  rolling window.
- `metrics.csv`: baseline, candidate validation và test cuối.

Split manifest nằm trong `reports/manifests/dataset_manifest.json`.

### 3. Re-evaluate hoặc inference

```powershell
python -m src.models.evaluate_models
python -m src.models.inference --target rating2_all --input request.csv --output predictions.csv
```

`request.csv` có thể là full model feature rows hoặc sparse future-map rows
chứa `map`, `agent`, `team`, `opponent_team`. Encoder unknown và imputer cho
phép unseen player/agent không làm inference lỗi.

## Quy tắc leakage

Model không nhận trực tiếp `player_id`, `team_id`, `opponent_team_id`,
`match_id`, `event_id`, targets, round scores, `map_winner` hoặc thống kê được
tạo sau map. ID chỉ được dùng để tra lịch sử. Mọi rolling history được
snapshot trước map rồi mới update sau map. Categorical fields biết trước map
gồm map, agent, team, opponent team, team pick và optional
`event_stage`/`event_round` nếu nguồn dữ liệu cung cấp. Feature player mới gồm
EWMA 3/5/10 map, form-vs-long-term gap, context player-agent/player-map,
opponent strength theo thời điểm và số trận trong 7/14/30 ngày.

Schema nằm trong `src/features/feature_schema.py`; logic lịch sử nằm trong
`src/features/historical_features.py`.

## Kiểm thử

```powershell
python -m pytest -q
```

Tests kiểm tra duplicate key, split isolation, historical feature không dùng
tương lai, leakage schema và inference với category chưa biết.

## Crawl nhiều khu vực

Runner crawl tuần tự EMEA, Americas và Pacific, lưu checkpoint tại
`data/raw/crawl_checkpoint.json`, audit từng khu vực và merge thành các file
`all_regions_*.csv`:

```powershell
python run_pipeline.py --regions EMEA AMERICAS PACIFIC --skip-modeling
```

Có thể chạy thử một khu vực trước:

```powershell
python run_pipeline.py --regions EMEA --skip-modeling
```

Chạy lại stage đã hoàn tất bằng `--force`. Mỗi crawler ghi file theo mẫu
`<REGION>_<dataset>.csv`; audit nằm trong `<REGION>_raw_audit.json` và cột
`region` được giữ lại trong dữ liệu merge để phân tích và train model.

## Cấu trúc chính

```text
src/preprocessing/build_player_match_dataset.py
src/features/feature_schema.py
src/features/historical_features.py
src/models/train_models.py
src/models/evaluate_models.py
src/models/inference.py
data/processed/
models/
reports/modeling/
reports/manifests/
tests/test_pipeline.py
```
