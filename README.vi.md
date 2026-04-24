# VN30 chaos-Bayes-LSTM — tài liệu tiếng Việt

Hướng dẫn chạy code và đọc dữ liệu. Phiên bản gọn.

## Yêu cầu

- Ubuntu 22.04 / Debian 12 / WSL.
- Python 3.10 trở lên.
- Dung lượng trống: 2 GB.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy demo

```bash
jupyter notebook notebooks/demo_vcb_chaos_bayes_lstm.ipynb
```

Notebook chạy end-to-end trên một mã VCB. Đầu ra gồm: chỉ báo hỗn loạn, kiểm định surrogate, dự báo LSTM, trung bình Bayes.

## Bố cục thư mục

| Đường dẫn | Nội dung |
|-----------|----------|
| `src/python/` | chaos indicators, IAAFT, LSTM, BMA |
| `src/R/` | cross-check Lyapunov bằng nonlinearTseries |
| `src/scripts/` | CLI drivers để chạy full pipeline |
| `notebooks/` | demo VCB end-to-end |
| `data/` | parquet đã xử lý (~2.5 MB) |

## Dữ liệu

File `data/vn30_panel_1D.parquet` (hoặc `.csv` / `.xlsx`):

| Cột | Kiểu | Ý nghĩa |
|-----|------|--------|
| `time` | datetime | mốc phiên, múi giờ Việt Nam |
| `ticker` | string | mã cổ phiếu VN30 |
| `close` | float | giá đóng cửa, đơn vị nghìn đồng |
| `ret_log` | float | log-return theo ngày |
| `parkinson_vol` | float | biến động Parkinson theo phiên |

Khoảng dữ liệu: 2021-01-05 tới 2025-12. Nguồn: VCI + KBS qua `vnstocks`.

## Mở file trong Excel

Mở trực tiếp `vn30_panel_1D.xlsx`. 36 243 dòng × 5 cột. Không cần công thức.

## Quy trình

1. Kéo dữ liệu thô: `src/scripts/pull_daily_long.py` + `pull_5m_intraday.py`.
2. Tiền xử lý: `src/python/data_prep.py`.
3. Chỉ báo hỗn loạn: `src/python/chaos_indicators.py`.
4. Kiểm định null: `src/python/surrogates.py` (IAAFT).
5. Dự báo: `src/python/lstm_ensemble.py` (MC dropout, 20 lần).
6. Tổng hợp Bayes: `src/python/bma.py`.

## Giấy phép

MIT. Xem `LICENSE`.

## Trích dẫn

Xem BibTeX trong `README.md` (bản tiếng Anh).
