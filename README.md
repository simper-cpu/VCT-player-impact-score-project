# VCT Player Impact Score Project

Forecast VCT players' impact score and KDA.

## Project structure

```text
data/               Raw, processed, and external datasets
notebooks/          Exploratory analysis notebooks
src/collection/     Data collection scripts
src/preprocessing/  Data cleaning and transformation
src/features/       Feature engineering
src/models/         Model training and evaluation code
src/visualization/  Charts and result visualizations
models/             Saved trained models
reports/figures/    Exported charts and figures
tests/              Automated tests
```

## Crawl VLR events

Install dependencies, then run:

```powershell
python src/collection/crawl_events.py
```

The collected data is saved to `data/raw/vlr_events.csv`.
