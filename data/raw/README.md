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

Each zip holds `sample_orig_YYYY.txt` (origination) and `sample_perf_YYYY.txt` (monthly performance),
in the Release 48 layout. Older downloads, which name the performance file `sample_svcg_YYYY.txt`, also
work. Extracted `.txt` files anywhere under this folder also work.

To get them, sign in at https://claritydownload.fmapps.freddiemac.com/CRT/ (register first if needed) and
accept the dataset terms. Open the **SFLLD Data** tab, not "CRT Data", and download the `sample_YYYY.zip`
files. Each holds 50,000 loans and is 30–55 MB. The multi-GB `historical_data_*` files are the full dataset
and aren't needed.

The terms do not allow redistribution, so `.gitignore` keeps everything in this folder (except this note) and
everything in `data/processed/` and `models/` out of version control.
