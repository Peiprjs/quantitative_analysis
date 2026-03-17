# DACIL-WESENSE Quantitative Analysis

End-to-end pipeline for CPET (cardiopulmonary exercise testing) data collected
with the WESENSE sensor system.  Given a set of patient folders it produces
cleaned telemetry CSVs, exploratory plots, unsupervised ML visualisations, and
a COPD exacerbation risk score for each patient.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Input data format](#input-data-format)
3. [Quickstart](#quickstart)
4. [Step-by-step usage](#step-by-step-usage)
   - [Convert XLS files to CSV](#1-convert-xls-files-to-csv)
   - [Convert BDF files to CSV/TSV](#2-convert-bdf-files-to-csvtsv)
   - [Run the main analysis pipeline](#3-run-the-main-analysis-pipeline)
   - [Run the COPD risk scoring notebook](#4-run-the-copd-risk-scoring-notebook)
   - [Run the Streamlit GUI](#5-run-the-streamlit-gui)
5. [Configuration reference](#configuration-reference)
6. [Output reference](#output-reference)
7. [COPD risk scoring](#copd-risk-scoring)
8. [Streamlit GUI](#streamlit-gui)

---

## Prerequisites

**Conda environment (recommended)**

```bash
conda create -n quantitative_analysis python=3.13
conda activate quantitative_analysis
pip install -r requirements.txt
```

**Jupyter kernel registration** (run once after creating the env)

```bash
python -m ipykernel install --user --name quantitative_analysis --display-name "quantitative_analysis"
```

Then open `main.ipynb` or `copd_risk.ipynb` and select the
`quantitative_analysis` kernel.

---

## Input data format

### Folder structure

Place one sub-folder per patient/trial inside `./data`.  The folder name
becomes the patient identifier used throughout the pipeline and in all output
filenames.

```
data/
└── Patient 1/                          ← any name; becomes the patient ID
    ├── WESENSET<id>.xls                ← CPET telemetry export (tab-separated)
    ├── WESENSET<id>.csv                ← produced by xls_to_csv.py (see below)
    ├── WESENSETEST_<id>_L1_ECG*.bdf    ← ECG recording, sensor L1 (optional)
    └── WESENSETEST_<id>_L2_ECG*.bdf    ← ECG recording, sensor L2 (optional)
```

Multiple patient folders are supported:

```
data/
├── Patient 1/
├── Patient 2/
└── Patient 3/
```

Any extra files in a patient folder (`.dat`, `.log`, etc.) are ignored.

### CPET telemetry file (`.xls` / `.csv`)

The raw export from the WESENSE system is saved as `.xls` but is actually a
**tab-separated text file** encoded in ISO-8859-1.  Run `xls_to_csv.py` to
convert it (see [step 1](#1-convert-xls-files-to-csv)).

After conversion the CSV has the following structure:

```
Row 0  Identificatie: | <id>  | Naam: | <name> | Voornaam: | <first>
Row 1  Bezoekdatum:   | <date>| Geboortedatum: | <dob> | Leeftijd: | 29 Jaar
Row 2  Geslacht:      | vrouw | Gewicht: | 68.0 kg | BMI: | 24 kg/m²
Row 3  Gebruiker:     | ...
Row 4  (empty)
Row 5  Tijd | Belasting | RPM | V'E | BF | VTex | V'O2 | V'CO2 | RER | HR | …  ← column headers
Row 6  min  | W         | … (units row — skipped automatically)
Row 7+ <time-series data>
```

Stage annotation rows are interspersed in the data.  Each annotation row
contains one of the following keywords and labels all subsequent rows until
the next annotation:

| Keyword    | Meaning                 |
|------------|-------------------------|
| `Opwarmen` | Warm-up                 |
| `Test`     | Incremental test        |
| `VT1`      | Ventilatory threshold 1 |
| `VT2`      | Ventilatory threshold 2 |
| `Herstel`  | Recovery                |

The pipeline tolerates missing stages — if `VT1` or `VT2` rows are absent the
corresponding features will be `NaN` or flagged accordingly.

**Per-patient, the pipeline picks the first `.csv` file it finds in the
folder.**  If a folder contains multiple CSV files, rename or remove the ones
you do not want processed.

### BDF ECG files (optional)

BDF files are loaded with MNE.  Both L1 and L2 are optional — missing files
generate a warning and the pipeline continues without ECG features.

Naming pattern (case-insensitive):

```
WESENSETEST_<id>_L1_ECG<anything>.bdf
WESENSETEST_<id>_L2_ECG<anything>.bdf
```

---

## Quickstart

```bash
# 1. Activate the environment
conda activate quantitative_analysis

# 2. Place patient folders under ./data  (see Input data format above)

# 3. Convert raw .xls exports to CSV
python xls_to_csv.py

# 4. Open the main pipeline and run all cells
jupyter notebook cpet.ipynb

# 5. (Optional) Run the COPD risk scoring notebook
jupyter notebook copd_risk.ipynb

# 6. (Optional) Run the Streamlit GUI
streamlit run app.py
```

Results are written to `./output/`.

---

## Step-by-step usage

### 1. Convert XLS files to CSV

The WESENSE device exports tab-separated `.xls` files.  Convert them before
running either notebook:

```bash
# Convert all .xls files under ./data (skips files that already have a .csv)
python xls_to_csv.py

# Re-convert even if a .csv already exists
python xls_to_csv.py --overwrite

# Preview what would be converted without writing any files
python xls_to_csv.py --dry-run

# Use a different data directory
python xls_to_csv.py --data-dir /path/to/data
```

The script prints a summary:

```
  OK     data/Patient 1/WESENSET165653.xls -> data/Patient 1/WESENSET165653.csv
  SKIP   data/Patient 1/WESENSET144118.xls  (CSV exists; use --overwrite to re-convert)

Done. Converted 1, skipped 1, errors 0.
```

### 2. Convert BDF files to CSV/TSV

Biosemi BDF files (ECG recordings) can be converted to human-readable
timeseries files with `bdf_to_csv.py`:

```bash
# Convert all .bdf files under ./data to CSV (one file per BDF)
python bdf_to_csv.py

# Write tab-separated files instead
python bdf_to_csv.py --format tsv

# Place all output files in a single directory
python bdf_to_csv.py --output-dir ./output/ecg_raw

# Export only specific channels
python bdf_to_csv.py --channels Fp1 Fp2 Status

# Re-convert even if an output file already exists
python bdf_to_csv.py --overwrite

# Preview what would be converted without writing any files
python bdf_to_csv.py --dry-run
```

Each output file has one row per sample and the following columns:

| Column     | Contents                                                                |
|------------|-------------------------------------------------------------------------|
| `time_s`   | Elapsed time in seconds from the start of the recording                 |
| `<ch_name>`| Signal value in **µV** for EEG/ECG/EMG/EOG channels; native unit otherwise |
| `status`   | Integer trigger / event code from the BDF Status channel                |

The output file is placed alongside the source `.bdf` unless `--output-dir`
is specified.

### 3. Run the main analysis pipeline

Open `main.ipynb` in Jupyter and **edit the configuration cell** near the top:

```python
DATA_ROOT  = "data"    # path to the folder containing patient sub-folders
OUTPUT_ROOT = "output" # all generated files go here
```

Then **Run All** (`Kernel → Restart & Run All`).

The pipeline runs the following steps for every patient folder:

| Step           | What it does                                                               |
|----------------|----------------------------------------------------------------------------|
| Load telemetry | Reads the CSV, parses stage annotations, returns a cleaned DataFrame       |
| Load ECG       | Reads L1 / L2 BDF files with MNE; extracts mean amplitude and estimated HR |
| Sync           | Merges ECG features into the telemetry DataFrame                           |
| Save           | Writes `<patient_id>_telemetry.csv` to `output/<patient_id>/`              |
| Summarise      | Appends a row to the master summary                                        |

After batch processing, the notebook produces EDA plots (stage profiles,
V-slope, RER trajectory, HRR), a cross-patient correlation heatmap, and
unsupervised ML analysis (PCA, K-Means, DBSCAN).

Progress bars show throughput for each batch step.  Failed patients are
logged to `pipeline.log` and skipped — they do not interrupt the pipeline.

### 4. Run the COPD risk scoring notebook

Open `copd_risk.ipynb` and set the same two path variables:

```python
DATA_ROOT  = "data"
OUTPUT_ROOT = "output"
```

Then **Run All**.  For each patient it:

1. Extracts 9 CPET biomarkers from the telemetry (see [COPD risk scoring](#copd-risk-scoring))
2. Compares each marker against a clinical threshold
3. Assigns a composite risk level: **Low** (0 flags), **Moderate** (1–2 flags), **High** (≥ 3 flags)
4. Generates a radar chart, a cross-patient bar chart, and (if ≥ 3 patients) a PCA scatter

### 5. Run the Streamlit GUI

Run the app from the repository root:

```bash
streamlit run app.py
```

The GUI supports:
- selecting a patient folder
- viewing CPET telemetry and stage-based plots
- computing COPD features/risk
- loading available ECG BDF files (L1/L2) and showing extracted ECG features

---

## Configuration reference

### `main.ipynb`

| Variable          | Location            | Default                                         | Description                                                         |
|-------------------|---------------------|-------------------------------------------------|---------------------------------------------------------------------|
| `DATA_ROOT`       | Cell 2              | `"data"`                                        | Root directory containing patient sub-folders                       |
| `OUTPUT_ROOT`     | Cell 2              | `"output"`                                      | Base output directory                                               |
| `EDA_METRICS`     | EDA section         | `["HR", "VO2", "VCO2", "Power", "RER", "SpO2"]` | Metrics shown in per-patient stage plots                            |
| `CORR_CANDIDATES` | Correlation section | `["HR", "VO2", ...]`                            | Columns included in the pooled correlation heatmap                  |
| `N_CLUSTERS`      | K-Means cell        | `4`                                             | Number of K-Means clusters — adjust after inspecting the elbow plot |
| `eps`             | DBSCAN cell         | `1.0`                                           | DBSCAN neighbourhood radius (in standard-deviation units)           |
| `min_samples`     | DBSCAN cell         | `5`                                             | DBSCAN minimum neighbourhood size                                   |

### `copd_risk.ipynb`

| Variable      | Location | Default    | Description                                   |
|---------------|----------|------------|-----------------------------------------------|
| `DATA_ROOT`   | Cell 2   | `"data"`   | Root directory containing patient sub-folders |
| `OUTPUT_ROOT` | Cell 2   | `"output"` | Base output directory                         |

To change the clinical thresholds used for risk flagging, edit the
`_COPD_THRESHOLDS` dictionary in `functions.py` (see
[COPD risk scoring](#copd-risk-scoring)).

### `xls_to_csv.py`

| Flag          | Default  | Description                                      |
|---------------|----------|--------------------------------------------------|
| `--data-dir`  | `./data` | Directory to search recursively for `.xls` files |
| `--overwrite` | off      | Re-convert even if a `.csv` already exists       |
| `--dry-run`   | off      | Print what would be done without writing files   |

### `bdf_to_csv.py`

| Flag             | Default              | Description                                                              |
|------------------|----------------------|--------------------------------------------------------------------------|
| `--data-dir`     | `./data`             | Directory to search recursively for `.bdf` files                         |
| `--output-dir`   | same as source BDF   | Directory to write output files                                          |
| `--format`       | `csv`                | Output format: `csv` (comma-separated) or `tsv` (tab-separated)         |
| `--channels`     | all data channels    | Whitelist of channel names to include in the output                      |
| `--overwrite`    | off                  | Re-convert even if an output file already exists                         |
| `--dry-run`      | off                  | Print what would be done without writing files                           |

---

## Output reference

```
output/
├── pipeline.log                        ← combined log from all notebook runs
├── master_summary.csv                  ← one row per patient, all peak/mean metrics
├── batch_vo2_by_gender.png             ← box plot: VO2 distribution by sex
├── batch_bmi_vs_vo2.png                ← scatter: BMI vs peak VO2
├── scree.png                           ← PCA explained variance
├── pca_2d_stage.png                    ← PCA scatter coloured by exercise stage
├── elbow.png                           ← K-Means inertia vs K
├── kmeans_clusters.png                 ← K-Means cluster scatter
├── dbscan_clusters.png                 ← DBSCAN cluster scatter
│
├── <patient_id>/                       ← one sub-folder per patient
│   ├── <patient_id>_telemetry.csv      ← cleaned telemetry with Stage column
│   └── <patient_id>_metrics_by_stage.png
│
├── copd_risk_summary.csv               ← one row per patient: markers + flags + risk score
└── copd_risk/
    ├── <patient_id>_copd_radar.png     ← radar chart of normalised CPET markers
    ├── copd_risk_bar.png               ← flag-count bar chart across patients
    └── all_patients_pca_2d_risk.png    ← PCA scatter coloured by risk level (≥3 patients)
```

`pipeline.log` uses ISO 8601 timestamps and is appended on every run.  To
start a fresh log, delete the file before running.

---

## COPD risk scoring

`copd_risk.ipynb` derives a risk score from 9 CPET markers.  Each marker is
compared against the threshold below; a flag of `1` means the value is in the
range associated with elevated COPD risk.

| Marker                   | CSV column(s)                      | Flag condition | Threshold    | Source                          |
|--------------------------|------------------------------------|----------------|--------------|---------------------------------|
| Peak VO₂/kg              | `V'O2` ÷ weight                    | `< threshold`  | 20 mL/min/kg | ATS/ACCP 2003                   |
| VE/VCO₂ slope            | `V'E`, `V'CO2` (linear regression) | `> threshold`  | 34           | Puente-Maestu *et al.* ERJ 2016 |
| SpO₂ nadir               | `SpO2`                             | `< threshold`  | 95 %         | ATS/ERS                         |
| SpO₂ drop                | baseline − nadir                   | `≥ threshold`  | 4 pp         | ATS/ERS                         |
| Peak breathing frequency | `BF`                               | `> threshold`  | 40 br/min    | Wasserman 5th ed.               |
| Peak RER                 | `RER`                              | `< threshold`  | 1.0          | Wasserman 5th ed.               |
| Peak O₂ pulse            | `VO2/HR`                           | `< threshold`  | 10 mL/beat   | Wasserman 5th ed.               |
| VT1 presence             | `Stage == "VT1"`                   | absent (= 0)   | —            | —                               |

**Composite score:**

| n\_flags | Risk level |
|----------|------------|
| 0        | Low        |
| 1–2      | Moderate   |
| ≥ 3      | High       |

To change any threshold, edit `_COPD_THRESHOLDS` in `functions.py`.  The
baseline SpO₂ is computed as the median of the first 10 data rows
(resting / warm-up phase).

> **Note:** This scoring is intended as a research screening tool, not a
> clinical diagnostic.  Thresholds are derived from group-level CPET
> reference values and should be validated against your specific population.

## Streamlit GUI

`app.py` is an interactive interface built on top of `functions.py`. It reads
the same patient data layout (`data/<patient>/`) and uses the same CPET/ECG
processing helpers as the notebooks.
