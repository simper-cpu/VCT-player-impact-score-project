# Hướng dẫn hiểu và tối ưu pipeline VCT

Tài liệu này giải thích pipeline hiện tại từ CSV thô đến model dự đoán `rating2_all`, `acs_all` và `kda_all`. Mục tiêu là giúp đọc được syntax Python/pandas/scikit-learn trong project, biết file nào nối với file nào, và biết nên tối ưu ở đâu mà không làm sai dữ liệu.

## 1. Bức tranh tổng thể

```text
VLR/web
  ↓ crawl
data/raw/*_events*.csv
data/raw/*_matches.csv
data/raw/*_players_stat.csv
data/raw/*_match_maps_stat.csv
  ↓ merge + làm sạch + kiểm tra khóa
data/processed/player_match_dataset_cleaned.csv
  ↓ tạo lịch sử trước map, Elo, form, context
data/processed/player_match_features.csv
  ↓ split theo thời gian + train + validate + test
models/vct_*_pipeline.joblib
reports/modeling/*.csv
  ↓ inference
predictions.csv
```

Ba tầng dữ liệu quan trọng:

| Tầng | Một dòng đại diện cho | Khóa |
|---|---|---|
| `matches` | Một match | `match_id` |
| `match_maps_stat` | Một map trong match | `(match_id, map)` |
| `players_stat` | Một player trong một map | `(player_id, match_id, map)` |

Dataset model cuối cùng có grain là **một player trong một map**. Vì vậy mọi bảng khác phải được đưa về đúng grain trước khi join.

## 2. Chạy project

Từ thư mục gốc `VCTProject`:

```powershell
python -m pip install -r requirements.txt

# Crawl nhiều khu vực, không train model
python run_pipeline.py --regions EMEA AMERICAS PACIFIC --skip-modeling

# Làm sạch và tạo dataset player-map
python src/preprocessing/build_player_match_dataset.py

# Train, validate, tạo report và export model
python -m src.models.train_models

# Chạy test
python -m pytest -q

# Kiểm tra syntax/import
python -m compileall -q app.py src
```

Muốn crawl lại một stage đã hoàn thành:

```powershell
python run_pipeline.py --regions EMEA --force --skip-modeling
```

`crawl_checkpoint.json` giúp tiếp tục crawl dang dở. `--force` sẽ chạy lại stage tương ứng, nên chỉ dùng khi cần làm mới dữ liệu.

## 3. Crawl tạo ra những file gì?

Ví dụ:

```text
data/raw/AMERICAS_matches.csv
data/raw/EMEA_matches.csv
data/raw/PACIFIC_matches.csv
data/raw/all_regions_matches.csv
```

`run_pipeline.py` đọc file từng vùng, thêm cột `region` nếu thiếu rồi `pd.concat` theo chiều dọc. Đây là phép **union/append**, không phải join theo cột. Sau đó script loại duplicate theo khóa phù hợp, ví dụ `matches` dùng `match_id`, `match_maps_stat` dùng `(match_id, map_name)`.

Một số event quốc tế xuất hiện trong nhiều regional catalog. Code có audit để phân biệt duplicate hợp lệ và duplicate đáng ngờ. Không nên tự tay xóa các dòng overlap trước khi đọc `cross_region_audit.json`.

### Vì sao crawl có retry, worker và checkpoint?

- `workers`: số request chạy song song; tăng quá cao có thể bị rate-limit.
- `retries`: số lần thử lại nếu request lỗi.
- `checkpoint`: ghi stage đã hoàn thành để không crawl lại toàn bộ.
- `completed_matches_only`: mặc định bỏ match tương lai/TBD vì chưa có player/map stats.

## 4. Cách đọc syntax Python trong project

### `Path`

```python
ROOT = Path(__file__).resolve().parents[2]
input_path = ROOT / "data" / "raw" / "all_regions_players_stat.csv"
```

`Path / "folder"` nối đường dẫn theo cách chạy được trên Windows và Linux. `__file__` là file Python hiện tại.

### DataFrame và Series

```python
df = pd.read_csv("input.csv")      # DataFrame: bảng
df["acs_all"]                      # Series: một cột
df[["player_id", "match_id"]]     # DataFrame: nhiều cột
```

Một số lệnh kiểm tra cơ bản:

```python
df.shape
df.columns.tolist()
df.head()
df.dtypes
df.isna().sum().sort_values(ascending=False)
df["match_id"].nunique()
df.duplicated(["player_id", "match_id", "map"]).sum()
```

### `loc`, `dropna`, `drop_duplicates`

```python
df.loc[df["team_id"].notna()]
df.dropna(subset=["match_id", "team_id"])
df.drop_duplicates(["player_id", "match_id", "map"], keep="last")
```

- `loc[điều_kiện]`: lọc dòng.
- `dropna(subset=...)`: bỏ dòng thiếu cột bắt buộc.
- `drop_duplicates(key)`: giữ một dòng cho mỗi khóa.

Không nên dùng `drop_duplicates()` không chỉ rõ cột vì hai dòng có thể khác một cột phụ nhưng vẫn là cùng một player-map.

### Ép kiểu, hàm và `groupby`

```python
df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

team_mean = df.groupby("team_id")["rating2_all"].mean()
```

`errors="coerce"` biến giá trị lỗi thành `NaN/NaT`, sau đó cần kiểm tra số lượng lỗi. `groupby` là gom theo nhóm. Khi có thể, ưu tiên `np.where`, `merge`, `groupby` thay vì `apply(axis=1)` vì thường nhanh hơn.

## 5. Cách join các CSV đúng

### `concat` và `merge` khác nhau

`concat` nối các bảng cùng ý nghĩa theo chiều dọc:

```python
all_regions = pd.concat([americas, emea, pacific], ignore_index=True)
```

`merge` nối thêm cột theo khóa:

```python
dataset = players.merge(matches, on="match_id", how="left", validate="many_to_one")
dataset = dataset.merge(
    maps, on=["match_id", "map"], how="left", validate="many_to_one"
)
```

`players` có nhiều dòng cho một `match_id`, còn `matches` chỉ có một dòng, nên quan hệ đúng là `many_to_one`. `maps` phải có tối đa một dòng cho mỗi `(match_id, map)`.

### Join hiện tại của project

Trong `build_player_match_dataset.py`:

1. Đọc `all_regions_players_stat.csv`.
2. Chuẩn hóa numeric và phần trăm.
3. Tạo `kda_all = (kills_all + assists_all) / deaths_all`.
4. Loại duplicate player-map theo `(player_id, match_id, map)`.
5. Đọc `all_regions_matches.csv`, giữ thông tin match/event/date/time.
6. Đọc `all_regions_match_maps_stat.csv`, đổi `map_name` thành `map`.
7. Loại duplicate map theo `(match_id, map)`.
8. Join player stats với match info bằng `match_id`.
9. Join với map info bằng `(match_id, map)`.
10. Tạo `opponent_team`, Elo trước map và giữ các cột hợp lệ.

```text
players_stat (n dòng / match)
       -- match_id --> matches (1 dòng / match)
       -- match_id + map --> match_maps_stat (1 dòng / map)
                                ↓
                  player_match_dataset_cleaned
```

### Kiểm tra trước và sau join

```python
players = pd.read_csv("data/raw/all_regions_players_stat.csv", low_memory=False)
matches = pd.read_csv("data/raw/all_regions_matches.csv", low_memory=False)
maps = pd.read_csv("data/raw/all_regions_match_maps_stat.csv", low_memory=False)

print(players.shape, matches.shape, maps.shape)
print(players["match_id"].isna().sum())
print(matches["match_id"].duplicated().sum())
print(maps.duplicated(["match_id", "map_name"]).sum())

assert not matches.duplicated("match_id").any()
assert not maps.duplicated(["match_id", "map_name"]).any()

joined = players.merge(
    matches[["match_id", "event_id", "match_date", "match_time"]],
    on="match_id", how="left", validate="many_to_one",
)
```

Nếu `validate="many_to_one"` lỗi, bảng bên phải bị duplicate khóa. Nếu sau `left join` số dòng tăng, gần như chắc chắn bảng bên phải có duplicate key hoặc join sai grain.

### Chọn kiểu join

- `left`: giữ mọi player-map; phù hợp cho dataset model.
- `inner`: chỉ giữ dòng có match ở cả hai bên; dễ làm mất dữ liệu âm thầm.
- `outer`: dùng để audit record chỉ xuất hiện ở một nguồn.

Đừng dùng `inner` chỉ để làm hết `NaN`; đó là che giấu vấn đề crawl/join.

### Join theo ID, không theo tên

Ưu tiên `match_id`, `(match_id, map)`, `player_id`, `team_id`. Tên map/team/player có thể đổi format và chỉ nên dùng để hiển thị.

## 6. Làm sạch dữ liệu và target

Target của project là biến cần dự đoán trước map:

- `rating2_all`: rating tổng của player trong map;
- `acs_all`: ACS tổng;
- `kda_all`: `(kills + assists) / deaths`.

Các chỉ số như kills, deaths, ACS, rating, round score, `map_winner` là kết quả sau map. Chúng được giữ trong cleaned dataset để phân tích nhưng không được đưa trực tiếp vào `X` của model.

Phần trăm như `83%` được đổi thành `0.83`. KDA có thể thiếu khi deaths bằng 0 vì code thay mẫu số 0 bằng missing; không nên tự thay bằng một số tùy ý nếu chưa quyết định rõ ý nghĩa thống kê.

## 7. Elo, historical features và data leakage

Leakage xảy ra khi model nhìn thấy thông tin chỉ biết sau map hoặc thông tin của tương lai. Ví dụ sai:

```python
df["team_avg_rating"] = df.groupby("team")["rating2_all"].transform("mean")
```

Dòng trên dùng cả rating của chính map và map tương lai.

Code Elo hiện tại duyệt map theo timestamp: lấy Elo hiện tại trước map, lưu `team_elo`, `opponent_elo`, `elo_gap`, rồi chỉ cập nhật sau khi biết winner. Đây là cách đúng cho feature pre-map.

`build_historical_features` tạo form player last/last 3/5/10, EWMA, long-term average, team/opponent history, map/agent/role history, rest days, số match 7/14/30 ngày, roster continuity, role composition, synergy và cold-start/fallback. History được snapshot trước map rồi mới update; các map cùng timestamp được xử lý theo batch.

Quy tắc đơn giản: với mỗi feature mới, hỏi “Ngay trước map bắt đầu, tôi có biết giá trị này không?”. Nếu không, không dùng.

## 8. Feature và model

Categorical gồm `map`, `agent`, `role`, `team`, `opponent_team`, `team_pick`, `event_stage`, `event_round`, `region`, `form_fallback_level`. Numeric gồm lịch sử và `match_year`, `match_month`, `match_dayofweek`.

ID như `player_id`, `team_id`, `match_id`, `event_id` không đưa trực tiếp vào model; ID chỉ dùng để tra lịch sử.

Pipeline scikit-learn có dạng:

```text
X → ColumnTransformer
      ├─ numeric: median imputation + missing indicators
      └─ categorical: most-frequent imputation + one-hot encoding
   → regressor
```

`OneHotEncoder(handle_unknown="ignore")` giúp inference không lỗi với agent/team/category mới. Imputer nằm trong pipeline nên được fit trên train, không phải toàn dataset.

Candidate hiện tại gồm Random Forest, XGBoost, XGBoost tuned, Huber/quantile cho KDA và CatBoost nếu đã cài. KDA còn thử `log1p` vì thường lệch phải. Không chọn model chỉ vì một metric trên một lần random split.

## 9. Split và đánh giá

Random split không phù hợp vì có thể đưa map tương lai vào train. Project chia theo timestamp và giữ toàn bộ dòng của cùng `match_id` trong cùng split:

```text
70% match cũ nhất: train
15% tiếp theo: validation
15% mới nhất: test cuối cùng
```

Rolling validation dùng train mở rộng theo thời gian, validate ở giai đoạn kế tiếp. Final test chỉ mở một lần ở cuối.

Metrics:

- `MAE`: sai số tuyệt đối trung bình, dễ hiểu.
- `RMSE`: phạt lỗi lớn mạnh hơn MAE.
- `R2`: mức giải thích variance, có thể âm.
- `Spearman`: chất lượng thứ hạng dự đoán.

Luôn so với global mean, player historical mean, rolling player mean và Elo/team-history baseline.

## 10. Đọc các report

### `metrics.csv`

```python
metrics = pd.read_csv("reports/modeling/metrics.csv")
print(metrics.query("target == 'acs_all'").sort_values(["split", "mae"])[
    ["model", "split", "mae", "rmse", "r2", "spearman"]
])
```

### `rolling_selection.csv`

Xem `mae`, `mae_std` và `mae_improvement_vs_global_mean`. Ưu tiên MAE thấp, dao động vừa phải và improvement dương.

### `*_test_predictions.csv`

Dùng để tìm nhóm model yếu:

```python
errors = pd.read_csv("reports/modeling/acs_all_test_predictions.csv")
print(errors.groupby("experience_bucket")["abs_error"].mean())
print(errors.groupby("map")["abs_error"].mean().sort_values(ascending=False))
```

### Feature importance

`*_feature_importance_raw.csv` là importance sau one-hot; `*_feature_importance_grouped.csv` dễ đọc hơn; `*_permutation_importance.csv` đo ảnh hưởng khi xáo trộn cột. Importance không chứng minh quan hệ nhân quả.

## 11. Quy trình tối ưu nên làm

### Bước 1: khóa dataset và kiểm tra chất lượng

```python
df = pd.read_csv("data/processed/player_match_dataset_cleaned.csv", low_memory=False)
print(df.shape)
print(df[["player_id", "match_id", "map"]].isna().sum())
print(df.duplicated(["player_id", "match_id", "map"]).sum())
print(df["match_date"].min(), df["match_date"].max())
print(df[["rating2_all", "acs_all", "kda_all"]].describe())
```

Nếu duplicate, missing key hoặc date sai, sửa preprocessing trước khi tuning.

### Bước 2: kiểm tra target

```python
for col in ["rating2_all", "acs_all", "kda_all"]:
    print(col, df[col].quantile([.01, .25, .5, .75, .99]).to_dict())
```

Xác minh outlier là lỗi crawl hay trận đấu thật trước khi xóa.

### Bước 3: thêm feature pre-map

Ưu tiên form gần đây, map/agent history, Elo, rest, roster và opponent strength. Tránh round score, winner, kills hoặc target của map hiện tại.

### Bước 4: tune có kiểm soát

Chỉ fit trên train, chọn bằng validation/rolling, giữ final test chưa đụng tới và ghi lại config. Không tune theo final test.

### Bước 5: đánh giá theo nhóm

Kiểm tra cold-start, agent/map hiếm, roster đổi và opponent chưa gặp bằng các report `errors_by_*`. Nhóm ít dòng không đủ để kết luận mạnh.

### Bước 6: kiểm tra stability

Nếu model thắng một window nhưng thua nhiều window khác, ưu tiên model ổn định hơn hoặc bổ sung dữ liệu/feature thay vì tăng độ phức tạp.

## 12. Inference

Nếu CSV đã có toàn bộ `MODEL_FEATURES`:

```powershell
python -m src.models.inference `
  --target rating2_all `
  --input request.csv `
  --output predictions.csv
```

Với sparse future-map rows và lịch sử map đã hoàn thành:

```powershell
python -m src.models.inference `
  --target acs_all `
  --input future_maps.csv `
  --history completed_maps.csv `
  --output acs_predictions.csv
```

`state_builder.py` tạo state trước map bằng cùng logic lịch sử lúc train. Không xây feature inference bằng kết quả của future map.

## 13. Lỗi thường gặp

### `KeyError` khi merge

Tên cột khác, ví dụ `map_name` và `map`. Rename trước join:

```python
maps = maps.rename(columns={"map_name": "map"})
```

### Số dòng tăng sau merge

Bảng bên phải bị duplicate key:

```python
maps[maps.duplicated(["match_id", "map"], keep=False)]
```

### Metric quá đẹp

Kiểm tra leakage, target/post-map columns, aggregate dùng toàn bộ dataset và split theo match.

### Category mới khi inference

`handle_unknown="ignore"` tránh crash, nhưng prediction cold-start vẫn kém tin cậy. Cần theo dõi fallback và sample size.

### Train chậm/hết RAM

Giảm `n_estimators`/`n_jobs`, chỉ giữ cột cần thiết, không load nhiều archive cùng lúc và chạy `--rf-only` để kiểm tra nhanh.

## 14. Cải tiến nên ưu tiên

1. Thêm `validate="many_to_one"` vào merge trong preprocessing.
2. Tạo quality report thống nhất: row count, unique key, missing key, unmatched join rate và date range.
3. Lưu history sample size/confidence cùng prediction để phân biệt cold-start.
4. Hyperparameter search có giới hạn trên rolling validation, không tune final test.
5. Thử prediction interval/quantile nếu cần độ bất định.
6. Theo dõi drift theo mùa giải, patch, region, map pool và agent meta.
7. Khi thêm feature, cập nhật `FEATURE_SCHEMA_VERSION`, manifest và test chống leakage.

## 15. Checklist trước khi tin vào model mới

```text
[ ] Dataset không duplicate (player_id, match_id, map)
[ ] Key không missing ngoài mức đã giải thích
[ ] matches unique theo match_id
[ ] map stats unique theo (match_id, map)
[ ] Merge không làm tăng dòng bất thường
[ ] Tất cả feature đều có trước map
[ ] Split theo thời gian và whole-match
[ ] Có global/player-history baseline
[ ] Model thắng trong rolling validation, không chỉ final test
[ ] Đã đọc lỗi theo experience/map/agent/roster
[ ] Đã lưu model manifest và feature schema version
[ ] Inference dùng đúng preprocessing bundle
```

## 16. Tóm tắt

- Dataset model là một dòng cho một player trong một map.
- `concat` gộp các region cùng loại; `merge` nối bảng bằng khóa.
- Join chính là `players --match_id→ matches` và `players --(match_id,map)→ map stats`.
- Luôn kiểm tra uniqueness ở phía phải của merge.
- Historical features và Elo phải được tính tuần tự trước map.
- Không dùng target, round score, winner hoặc stats của map hiện tại làm input.
- Chọn model bằng time split và rolling validation; xem metric cùng baseline.
- Tối ưu data quality và leakage thường quan trọng hơn chỉ tăng độ phức tạp model.
