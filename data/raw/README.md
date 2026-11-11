# Raw data (not committed)

Put the Freddie Mac **Single-Family Loan-Level Dataset** sample files here, as downloaded:

```
data/raw/sample_2014.zip
data/raw/sample_2015.zip
data/raw/sample_2016.zip
data/raw/sample_2017.zip
data/raw/sample_2018.zip   # optional, used for monitoring
data/raw/sample_2019.zip
data/raw/sample_2020.zip   # optional, and any later vintages: monitoring only
```

Each zip holds `sample_orig_YYYY.txt` (origination) and `sample_svcg_YYYY.txt` (monthly performance).
Extracted `.txt` files anywhere under this folder also work.

How to get them: register (free) on Freddie Mac's Clarity data site, accept the dataset terms, and download the
"Sample Dataset" files from the Single-Family Loan-Level Dataset page.

The terms do not allow redistribution, so `.gitignore` keeps everything in this folder (except this note) and
everything in `data/processed/` and `models/` out of version control.
