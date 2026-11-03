# The pipeline code

This folder holds the code of each stage of the pipeline as it ran on Google Colab, one folder per stage, with each module's original comments replaced by a short description (see Rerunning a stage). Each folder is the import root of its stage: the stage's driver and the packages it imports sit side by side in the layout that ran, and the folder `config/` holds the download plan or the run configuration that the stage read. The stages from 6 on loaded the shared folder `6_shared` first and then their own folder over it, so that a file of the stage folder took the place of the file at the same path in `6_shared`; the evaluation loaded `6_cpu_baselines` between the two. Each module opens with a one- or two-sentence description, which the tables at the end of this page repeat.

Some names recur across the folders. The packages `choi` (the core networks, loss, metrics and inference) and `choi_dl` (the download helpers) keep the names they had when the stages ran. The folders named `vendor/` hold modules of this pipeline, not third-party code: `3_drive_extraction/vendor/` holds copies of modules of `choi` and `choi_dl`, and `6_shared/train/vendor/` the comparison of two devices' results. The part TIER_A (tier_a in `tier_a_rules.py`) of the names `TIER_A_CONFIG.json`, `tier_a_rules.py` and `results/grid/MODELS_TIER_A.json` marks the trainings, baselines and evaluation that this repository reports: `TIER_A_CONFIG.json` is their configuration, `tier_a_rules.py` the drivers' rules module, and `MODELS_TIER_A.json` the table of models that the evaluation read.

Only the code is held here. The data are public at their sources: the EPA RSIG web service, the CMAS bucket of EQUATES on Amazon Web Services, the CMAS Google Drive folder of EQUATES, and Natural Earth. In the code, the test days of each held-out year are called `TEST_CV`, the held-back days the lockbox, and the sealed files are the predictions and metrics written for the evaluation. The package versions pinned for the pipeline, and which of them were recorded when the stages ran, are listed in `../environment/requirements-pipeline.txt`.

## The stages in run order

The first three stages are independent of one another and ran on separate runtimes; the arrays stage waited for the first two, and the fill stage for the third. Each stage below names what it read, what it wrote, the runtime it ran on and its recorded run time.

### `1_rsig_download`: the RSIG download

- Read: The RSIG download plan (`config/S2A_PLAN_RSIG.json`) and the EPA RSIG web service.
- Wrote: 677 compressed netCDF answers, each one month of daily aggregates over the UTC day on the 12 km grid: CMAQ's surface NO2 (the target) and the meteorological fields of 2015 to 2019, the lag day of 31 December 2014 and a second daily NO2 series for 2016. Each answer has a provenance row with its source, size and sha256, and the driver compared every answer's hash with the hashes of an earlier download of the same requests.
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: 1.56 h (24 October 2026).

### `2_aws_extraction`: the extraction from Amazon Web Services

- Read: The S3 download plan (`config/S2A_PLAN_S3.json`) and the public CMAS bucket of EQUATES on Amazon Web Services, read by anonymous ranged requests.
- Wrote: The 2016 MCIP grid, land-use and surface meteorology files; for each day of 2017 and 2019, the daily means of the merged gridded NO, NO2, HONO and VOC emissions and of the point and marine sectors summed into grid cells; and the two stack groups archives, each file with its provenance rows.
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: 10.94 h.

### `3_drive_extraction`: the extraction from the CMAS Google Drive

- Read: The CMAQ-ready emission archives of 2015, 2016 and 2018 from a copy of the CMAS EQUATES Google Drive folder, each archive pinned by its md5 and sha256 and tested as it was copied to the runtime.
- Wrote: For each day of the three years, the daily means of the gridded emissions and of the point and marine sectors summed into grid cells, with each stack's NO + NO2 + HONO sum, and a manifest and sums for each year, written after the year's checks and a value check by a separate decoder.
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: 3.53 h.

### `4_arrays`: the arrays

- Read: The outputs of stages 1 and 2, restored from the Drive; the Natural Earth states and provinces (lakes removed) and populated places, fetched by the code; and the two download plans.
- Wrote: The inventory of the verified inputs; the land mask (53,594 land cells of the 299 x 459 grid) and the cells of the ten cities; the input checks; the emission fields of 2017 and 2019 and their monthly weekday climatology; the splits, with the two-year emission tests (trained on 2017 and tested on 2019, and the reverse, each with the year's own emissions, its own emission climatology or the other year's); for each year 2015 to 2019, the input arrays (float16 and float32) and the NO2 target array in ppb, with the emission channels of 2015, 2016 and 2018 left as NaN for stage 5; arrays of climatological emissions for every year; the index files; the scaling statistics of the folds whose channels were complete; and the list of files that the later stages copy.
- Ran on: A high-memory Colab CPU runtime with about 75 GB of free disk.
- Recorded run time: The recorded run, a restore of 41.91 GB from the Drive followed by the 22 steps, lasted from 16:26:50 to 17:48:25 UTC on 25 October 2026.

### `5_fills_and_statistics`: the emission fills and the statistics

The folder also holds a screen of the NO2 target arrays, which the driver runs only when those arrays differ from an earlier build of the same arrays; they were equal, and the screen did not run.

- Read: The arrays and index files of stage 4, and the extracts of stage 3 for 2015, 2016 and 2018, each year once its check had passed.
- Wrote: The emission channels of 2015, 2016 and 2018, filled in place in the float32 input arrays, with their input checks and updated index files; then the scaling statistics of the five main folds.
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: The run that completed the stage recorded 0.05 h of driver time after its wait (04:45:38 to 04:48:45 UTC, 26 October 2026).

### `6_shared`: the shared package

The folder holds the shared package that every later stage loaded first: the core networks, loss, metrics and inference (`choi`), the download helpers (`choi_dl`), the trainer (`train`), the baselines (`baselines`), the rules of the training and evaluation runs, the trainer shim, and the training driver whose helpers the later drivers import.

### `6_cpu_baselines`: the baselines on a CPU runtime

- Read: The arrays and index files of stages 4 and 5, the splits and the held-back days, copied from the Drive with size checks.
- Wrote: A read-only check of every input array; the scaling statistics of the climatological-emission folds for 2016 and 2018 and of the two-year emission tests, written again without the held-back and buffer days; the cross-fitted climatology tables of each fold; and the sealed predictions of the baselines for the test days and the held-back days of each fold: the cell mean (CLIM-C), the weekday climatology (CLIM-WD), the emission-scaled climatology (CLIM-EM), the weather regression (MET-REG), the gradient-boosted trees (GBT), the linear model on the trees' features (GBT-RIDGE) and the trees stopped at their best validation round (GBT-ES).
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: 3.491 h.

### `6_cpu_norm_2019`: the statistics of one climatological-emission fold

- Read: The arrays, the splits and the held-back days.
- Wrote: The scaling statistics of the climatological-emission fold with 2019 held out, written again without the held-back and buffer days, with its index.
- Ran on: A high-memory Colab CPU runtime.
- Recorded run time: From 21:43 to 21:45 UTC on 26 October 2026.

### `7_train_reference`: the reference U-Net trainings

The folder holds the two files that differ from `6_shared`: the training driver and the rules module.

- Read: The arrays, index files, splits and scaling statistics of stages 4 to 6, from the Drive.
- Wrote: Eight trainings of the reference U-Net (P): the seed-0 training for each held-out year 2015 to 2019 and repeated trainings with other seeds (two with 2016 held out, one with 2019 held out), each with its checkpoints, training log and predictions of the test days. Their predictions of the held-back days were written later by the trainer of `7_train_residual`.
- Ran on: A Colab runtime with an NVIDIA A100 GPU (80 GB).
- Recorded run time: 3.114 h for the eight trainings, of which the training step took 175.9 min.

### `7_train_residual`: the residual U-Net and the other trainings

- Read: The same inputs as `7_train_reference`, with the cross-fitted climatology tables of `6_cpu_baselines`.
- Wrote: The other 43 trainings, in two lanes (two runtimes that ran side by side): the residual U-Net (M) for each held-out year, with repeated seeds with 2016 and with 2019 held out; the residual U-Net with climatological emissions (M-C7) for 2016, 2018 and 2019; the networks A1 to A4 of the steps between the two U-Nets for 2016 and 2019, and variants of A3 for 2016; the reference U-Net with the 18 inputs of Salman et al. (P18) for 2016; the two-year emission tests; the per-cell network (PIX-R) for each held-out year; and repeated seeds of the reference U-Net. Each has its checkpoints, training log and sealed predictions of the test days and the held-back days; the stage also wrote the held-back-day predictions of the eight trainings of `7_train_reference`. With those eight, the evaluation read 51 trained networks.
- Ran on: Colab runtimes with an NVIDIA A100 GPU (80 GB).
- Recorded run time: The two lanes ran on several runtimes between 19:27 UTC on 26 October and 17:33 UTC on 27 October 2026; for example, the residual U-Net with 2016 held out recorded a wall time of 2,747.5 s.

### `8_evaluation`: the evaluation

The evaluation package appears twice, in `eval/` and in `pe/code/eval/`, with the same content, as the stage's package held it. The module description of `eval/metrics.py` defines every metric, stratum and diagnostic, and a rerun of the evaluation writes that text, up to its last line, into each `metrics.json` as its `definitions` field.

- Read: The sealed predictions of the 51 networks and of the baselines, the target arrays, the land mask, the splits, the held-back days, the city cells, the channel index and the table of models.
- Wrote: The evaluation outputs held in `../results/`: first those of the test days of the five held-out years (`results/evaluation_test_years/`), then, behind a second gate, those of the held-back days (`results/evaluation_held_back/`).
- Ran on: A high-memory Colab CPU runtime (54.8 GB of memory).
- Recorded run time: 45.2 min for the test days and 72.1 min for the held-back days.

## Rerunning a stage

The driver of each folder is the entry point of its stage. A stage is rerun on a Colab runtime with the Google Drive mounted, after the outputs of the earlier stages are in place on the Drive: the stage folder is copied to the runtime (for a later stage, `6_shared` first and the stage folder over it, with `6_cpu_baselines` between the two for the evaluation), and the driver is run as a script with the runtime's Python, for example `python -I -B -u <driver>.py`. The drivers of stages 1 to 5 run their whole stage in one call, and a second call repeats only what is not yet done. The drivers of stages 6 to 8 take one step name as their one argument per call, in the order listed below, for example `python -I -B -u eval_colab_driver.py gates`. Before using an input, the drivers test it against the sizes and hashes that the earlier stages recorded, and they stop on any difference.

The drivers read and write under one Drive folder, `/content/drive/MyDrive/PROJECT_DATA/EMUL` (the constants `DRIVE` and `EMUL` near the top of each driver, and `SRC_ROOT` in `5_fills_and_statistics/b2c/copy_c7a_year.py`), and work on the runtime under `/content/emul`. Stage 3 reads its copy of the CMAS EQUATES Google Drive folder at the path `COPYDIR_INPUT` of `3_drive_extraction/c7a_colab_driver.py` and tests each archive against its md5 and sha256 pins before use. To rerun on another Drive, change these constants.

The modules in this folder carry a short description in place of the original comments, and some names in their messages were shortened for publication, so their file hashes differ from those of the files as they ran. Two modules check a file of this folder against a sha256 value they hold: `8_evaluation/eval/io.py` (and its copy in `8_evaluation/pe/code/eval/`) records whether the `train/guard.py` it imports is the expected file, and `6_shared/train/equiv_test.py` tests `train/vendor/compare_core.py` before use. The values they hold are the hashes of the files here, so both checks pass on this folder; the evaluation's records under `results/` keep the hashes of the files as they ran. The evaluation driver also checks each package it runs from against a list of its files' hashes (`CODE_SHA256SUMS_ASCII.txt`, whose hashes are set in `8_evaluation/config/TIER_A_CONFIG.json`); these lists belonged to the packages as they ran and are not included, so a rerun first writes them for the files here and sets their hashes in the configuration. A third check is left as it ran: `4_arrays/s3p2/p2_copy_cellsums.py`, which made `4_arrays/s3p2/p2_cellsums.py` from `c7a_reduce.py`, checks that file by its hash and line numbers as it ran, so it does not apply to the cleaned file. `CODE_AS_RUN.txt` gives, for every module, the sha256 of the file as it ran and a structure hash (the sha256 of its syntax tree without docstrings and with every string emptied) of both the file as it ran and the file here; `build_structure_hashes.py` recomputes the second for every module here (all are equal but `8_evaluation/nb9_gate.py`, where one identifier was renamed); the hashes were computed with Python 3.13, and the syntax-tree dump, and so the hash, differs between Python versions. Most fields named `decided_by` in the configurations and in `eval_run.json` hold a fixed identifier of one of the study's design decisions, R-01 to R-05; the code checks their form (R- followed by digits) and, in a few places, that they equal the identifier the code itself holds. The code's messages, configuration notes and provenance values keep some short step labels; the analyses and their order are in `plan/ANALYSIS_PLAN.md`.

| Folder | Driver | Steps, in order |
|---|---|---|
| `1_rsig_download` | `rsig_colab_driver.py` | the whole stage |
| `2_aws_extraction` | `c5_colab_driver.py` | the whole stage |
| `3_drive_extraction` | `c7a_colab_driver.py` | the whole stage |
| `4_arrays` | `p2_colab_driver.py` | the whole stage |
| `5_fills_and_statistics` | `s8_colab_driver.py` | the whole stage |
| `6_cpu_baselines` | `cpu_colab_driver.py` | `session`, `gates`, `copy`, `pd17`, `norm_f6`, `ctables`, `baselines`, `end` |
| `6_cpu_norm_2019` | `cpu_t2_driver.py` | `session`, `gates`, `copy`, `norm`, `end` |
| `7_train_reference` | `train_colab_driver.py` | `session`, `gates`, `copy`, `equiv`, `timing`, `env`, `train`, `end` |
| `7_train_residual` | `train2_colab_driver.py` | `session`, `gates`, `copy`, `equiv`, `timing`, `train`, `lockbox_pred`, `end` |
| `8_evaluation` | `eval_colab_driver.py` | `session`, `gates`, `copy`, `cv`, `lockbox`, `end` |

The step names stand for the following: `session` records the Colab session (its start, its hours and its processor) and does nothing else; `gates` checks the markers of the earlier stages, the configuration and the runtime before any work; `copy` copies the configuration files and the input arrays to the runtime, with size checks; `pd17` makes a read-only pass over every input array (its sha256 and its non-finite, fill and negative values) before any fit; `norm_f6` rewrites the scaling statistics of the climatological-emission folds of 2016 and 2018 and of the two-year emission tests with the held-back and buffer days left out; `ctables` builds the climatology tables of each fold; `baselines` refits the baselines on each fold's training days and writes their predictions; `norm` rewrites the scaling statistics of the climatological-emission fold with 2019 held out in the same way; `equiv` runs the equivalence test of the training code on the runtime's device; `timing` times the networks' training steps on the runtime's GPU, for the choice of GPU type; `env` checks that the runtime's GPU is of the type the timing step chose; `train` runs the trainings, each resumable from its checkpoints, and predicts the held-out year; `lockbox_pred` writes the held-back-day predictions of every finished training; `cv` evaluates the models on the test days of the five held-out years; `lockbox` evaluates the models on the held-back days, once their gate is open; and `end` records the compute used and what is done and what is left.

Some labels recur in the names of files, models and messages. F-6 (`f6`, and the models `A3_F-6_TRUE`, `A3_F-6_OTHER` and `A3_F-6_OWN`) is the two-year emission test: trained on 2017 and tested on 2019, and the reverse, each with the year's own emissions, its own emission climatology or the other year's. F-6a (`f6a`) is the climatological-emission folds, those of M-C7. C-7 (`c7`, and the files `emis_clim_*`) is the emission climatology, and C-7a (`c7a`) the extraction of each date's own emissions. The inputs carry the following channel labels, which are also keys of the results; T-1 is the target; in `4_arrays` T-2 is a check series (lstaconc DAILY_NO2 of 2016), neither an input nor a target; and in `3_drive_extraction` the alert ids T-2 and T-3 name two checks of that stage, events on the archive disk and the pace of the downloads.

| Label | Channel | Source | Unit |
|---|---|---|---|
| T-1 | NO2 (the target) | RSIG T1 aconc.NO2 DAILY_MEAN | ppb |
| M-1 | PRES layer 1 | RSIG M1 metcro3d.PRES L1 DAILY_MEAN | Pa |
| M-2 | SFC_TMP | RSIG M2 aconc.SFC_TMP DAILY_MEAN | C |
| M-3 | WSPD10 | RSIG M3 aconc.WSPD10 DAILY_MEAN | m/s |
| M-4 | UWIND layer 1, dot to cell | RSIG M4 metdot3d.UWIND L1 DAILY_MEAN, mean of the 4 corners | m/s |
| M-5 | VWIND layer 1, dot to cell | RSIG M5 metdot3d.VWIND L1 DAILY_MEAN, mean of the 4 corners | m/s |
| M-6 | SOL_RAD | RSIG M6 aconc.SOL_RAD DAILY_MEAN | W/m2 |
| M-7 | QV layer 1 | RSIG M7 metcro3d.QV L1 DAILY_MEAN | kg/kg |
| M-8 | PBLH | RSIG M8 aconc.PBLH DAILY_MEAN | m |
| M-9 | precip | RSIG M9 aconc.precip DAILY_MEAN | cm |
| L-1 | HT | MCIP GRIDCRO2D HT (static) | m |
| L-2 | LWMASK | MCIP GRIDCRO2D LWMASK (static) | 1 |
| L-3 | urban fraction | MCIP GRIDCRO2D PURB / 100 (static) | fraction |
| L-4 | ALBEDO | not in METCRO2D; built from GSW and RGRND for P18 only | - |
| L-5 | VEG | MCIP METCRO2D VEG, mean of records 00-23 UTC of the 15th of each month of 2016 (monthly) | fraction |
| L-6 | DLUSE | MCIP GRIDCRO2D DLUSE (static), the dominant land-use category code read as a number | 1 |
| E-1 | NO | merged layer-1 NO only, each year's own day; the emission climatology in inputs_c7 | mol/s |
| E-2 | NO2 | merged layer-1 NO2 only, each year's own day; the emission climatology in inputs_c7 | mol/s |
| E-3 | HONO | merged layer-1 HONO only, each year's own day; the emission climatology in inputs_c7 | mol/s |
| E-4 | VOC_INV | merged layer-1 VOC_INV, each year's own day; the emission climatology in inputs_c7 | g/s |
| P-1 | STACK NOx | point and marine NO + NO2 + HONO, each year's own day, the representative-day sectors by the weekday classes; the emission climatology in inputs_c7 | mol/s |
| X-1 | PBLH, previous day | M-8 of the day before | m |
| X-2 | WSPD10, previous day | M-3 of the day before | m/s |
| X-3 | UWIND, previous day | M-4 of the day before | m/s |
| X-4 | VWIND, previous day | M-5 of the day before | m/s |
| X-5 | SFC_TMP, previous day | M-2 of the day before | C |
| X-6 | PBLH daily maximum | RSIG X6 aconc.PBLH DAILY_MAX | m |

## The modules of each folder

### `1_rsig_download`

| Path | What it does |
|---|---|
| `choi_dl/__init__.py` | Marks the download package, which holds the shared helpers, the provenance rows, the progress files, and the clients and queues for the EPA RSIG web service and the public CMAS bucket on Amazon Web Services. |
| `choi_dl/common.py` | Helpers of the download code: the working folders and the record folder of logs, provenance and configuration, paths relative to the root folder, masking of text written to logs, paired UTC and local time stamps, line wrapping, and the temporary folder of the running process. |
| `choi_dl/dl_rsig.py` | Runs the queue of EPA RSIG requests in a fixed order: the CMAQ NO2 target and the meteorological series month by month for 2015 to 2019, the lag requests for 31 December 2014 and a second daily NO2 series for 2016. It skips items whose saved answer verifies, records and limits refusals, stops when disk runs short, and copies the provenance records. |
| `choi_dl/progress.py` | Writes the downloader's progress file (one ASCII line per item) and its status file of counts, replaced whole after each item; a write that fails is retried and then held for the next write, so that a busy file never stops the queue. |
| `choi_dl/prov.py` | Writes and reads the provenance rows of each downloaded file or extract (path, source, byte range, retrieval time, HTTP status, size, sha256, version and related fields) as a text record and a JSON-lines record per folder, verifies a file against its row, and copies the records to the record folder. |
| `choi_dl/rsig.py` | Client of the EPA RSIG web service: builds each monthly request, fetches it by paced streaming requests with three timed retries, treats any refusal or answer that holds no data as a stop, decompresses and checks the netCDF answer, and writes the file with its provenance row. |
| `config/S2A_PLAN_RSIG.json` | The RSIG download plan: the 677 requests in their order, with the coverage names and the sha256 of each coverage's description. |
| `rsig_colab_driver.py` | Driver of the RSIG download stage on Colab: runs the download queue on the RSIG plan (677 requests), restores earlier outputs from the Drive and mirrors new ones there every 120 s, and at the end compares each file's sha256 with the hashes kept from an earlier download of the same requests, writing the comparison as text and CSV. |

### `2_aws_extraction`

| Path | What it does |
|---|---|
| `c5_colab_driver.py` | Driver of the Amazon Web Services stage on Colab: runs the S3 download queue on its plan, which reads the public CMAS EQUATES bucket anonymously by ranged requests and writes the 2016 MCIP files, the daily emission extracts of 2017 and 2019 and the stack groups archives with their provenance rows, while restoring and mirroring its outputs on the Drive. |
| `choi/__init__.py` | Marks the core package: the networks, the IOA loss, the metrics, the whole-grid inference, the tar-header reader, the float16 storage and the stack-to-cell sums; importing it loads no heavy package. |
| `choi/stacks.py` | Sums point and marine emission stacks into grid cells: takes each stack's daily mean of NO, NO2 and HONO over the 24 hourly records and adds it to its cell of the 299 x 459 grid by the row and column in the stack groups file, counting the stacks outside the grid. |
| `choi/tario.py` | Reads tar headers by byte range without unpacking: member names (ustar, GNU long names and pax records), sizes, times, types and offsets, each header checksum verified, from any function that returns a byte range. |
| `choi_dl/__init__.py` | Marks the download package, which holds the shared helpers, the provenance rows, the progress files, and the clients and queues for the EPA RSIG web service and the public CMAS bucket on Amazon Web Services. |
| `choi_dl/common.py` | Helpers of the download code: the working folders and the record folder of logs, provenance and configuration, paths relative to the root folder, masking of text written to logs, paired UTC and local time stamps, line wrapping, and the temporary folder of the running process. |
| `choi_dl/dl_s3.py` | Runs the S3 queue against the public CMAS bucket in order: the 2016 MCIP files, the daily means of the merged gridded NO, NO2, HONO and VOC emissions for 2017 and 2019, the stack groups archives, and the point and marine sectors summed into cells, each written as a compressed NumPy file with its provenance row. |
| `choi_dl/progress.py` | Writes the downloader's progress file (one ASCII line per item) and its status file of counts, replaced whole after each item; a write that fails is retried and then held for the next write, so that a busy file never stops the queue. |
| `choi_dl/prov.py` | Writes and reads the provenance rows of each downloaded file or extract (path, source, byte range, retrieval time, HTTP status, size, sha256, version and related fields) as a text record and a JSON-lines record per folder, verifies a file against its row, and copies the records to the record folder. |
| `choi_dl/s3io.py` | Client of the public CMAS bucket on Amazon Web Services: anonymous HEAD and ranged GET requests to one host, paced and retried, with refusals never retried; it parses tar contents files, reads members inside the archives, opens HDF5 members, and computes daily grid means and per-stack means. |
| `config/S2A_PLAN_S3.json` | The S3 download plan: the names and ETags of the public objects and archive members read, the day sets, and the point and marine sectors kept. |

### `3_drive_extraction`

| Path | What it does |
|---|---|
| `c7a_colab_driver.py` | Driver of the Drive extraction stage on Colab: copies each pinned emission archive of 2015, 2016 and 2018 from a copy of the CMAS Google Drive folder to the runtime's disk, one at a time, tests its md5 and sha256 against the pins, runs the extraction on it, and copies each year's outputs to the Drive. |
| `c7a_common.py` | Fixed rules, places and helpers of the Drive extraction: archive kinds and families, the days of each year, the value sample, time stamps, hashing, text masking and wrapping, memory reads, and JSON and log writing. |
| `c7a_extract.py` | Runs the extraction of the 2015, 2016 and 2018 emission archives: plans each year's members from the tar headers, checks each archive, extracts the daily means member by member with manifest rows, and, when a year is complete, runs the year's checks and the separate value check. |
| `c7a_reduce.py` | Computes one tar member's daily means with the same reducers as the Amazon Web Services stage: gridded emissions as the mean of the 24 hourly records at layer 1, and point and marine stacks summed into cells, optionally with each stack's NO + NO2 + HONO sum, read either inside the archive or whole. |
| `c7a_tar.py` | Reads members inside a local tar without unpacking it: a seekable window over one member that records the byte ranges read and their sha256, a member listing by the tar-header reader, and the archive checks. |
| `c7a_valuecheck.py` | Checks the extracted values with a separate decoder that shares no code with the extraction: for a fixed sample of members it walks the tar, decodes the raw HDF5 chunks itself, recomputes the daily means and cell sums, and compares every array bit for bit with the extract. |
| `config/S2A_PLAN_S3.json` | The S3 download plan: the names and ETags of the public objects and archive members read, the day sets, and the point and marine sectors kept. |
| `ncclassic.py` | A minimal reader of netCDF classic files (CDF-1, CDF-2 and CDF-5) through any seekable file object, with the small part of the h5py interface that the reducers use, so that classic-format members can be read where netCDF4 is absent. |
| `vendor/choi/__init__.py` | Marks the core package: the networks, the IOA loss, the metrics, the whole-grid inference, the tar-header reader, the float16 storage and the stack-to-cell sums; importing it loads no heavy package. |
| `vendor/choi/stacks.py` | Sums point and marine emission stacks into grid cells: takes each stack's daily mean of NO, NO2 and HONO over the 24 hourly records and adds it to its cell of the 299 x 459 grid by the row and column in the stack groups file, counting the stacks outside the grid. |
| `vendor/choi/tario.py` | Reads tar headers by byte range without unpacking: member names (ustar, GNU long names and pax records), sizes, times, types and offsets, each header checksum verified, from any function that returns a byte range. |
| `vendor/choi_dl/__init__.py` | Marks the download package, which holds the shared helpers, the provenance rows, the progress files, and the clients and queues for the EPA RSIG web service and the public CMAS bucket on Amazon Web Services. |
| `vendor/choi_dl/common.py` | The download code's helpers (places, time stamps, masking and line wrapping), as copied for the Drive extraction. |
| `vendor/choi_dl/s3io.py` | The S3 client of the download code, as copied for the Drive extraction, which uses its tar and HDF5 readers and its grid and stack means. |

### `4_arrays`

| Path | What it does |
|---|---|
| `choi/__init__.py` | Marks the core package: the networks, the IOA loss, the metrics, the whole-grid inference, the tar-header reader, the float16 storage and the stack-to-cell sums; importing it loads no heavy package. |
| `choi/f16.py` | Float16 storage of the arrays: maps pressure from Pa to [0, 1] by (x - 50,000) / 60,000 before the cast and back, casts other channels plainly, and tests a channel's round trip, keeping it in float32 when it has non-finite values or a 99th-percentile relative error above 0.001. |
| `choi/infer.py` | Predicts the whole 299 x 459 grid from 64 x 64 windows at stride 32, blended by a two-dimensional Hann weight with a floor of 0.001 on an edge-padded grid, and returns the field in ppb with negative values set to zero and counted. |
| `choi/loss.py` | The training loss, 1 - IOA over the masked cells of a batch, computed as the ratio of the squared error to the potential error, with a guard that gives zero for a zero denominator. |
| `choi/metrics.py` | The evaluation metrics R, IOA, MAE, mean error, MSE and RMSE over paired emulated and CMAQ values, and the skill score against a reference, every sum in float64 with an optional mask; a non-finite pair raises an error. |
| `choi/nets.py` | The networks in PyTorch: a U-Net close to the published design, a U-Net with residual blocks, squeeze-and-excitation gates and dropout, and a per-pixel network, with helpers for widths, dropout and counts of parameters and multiply-accumulate operations. |
| `choi/stacks.py` | Sums point and marine emission stacks into grid cells: takes each stack's daily mean of NO, NO2 and HONO over the 24 hourly records and adds it to its cell of the 299 x 459 grid by the row and column in the stack groups file, counting the stacks outside the grid. |
| `choi/tario.py` | Reads tar headers by byte range without unpacking: member names (ustar, GNU long names and pax records), sizes, times, types and offsets, each header checksum verified, from any function that returns a byte range. |
| `choi_dl/__init__.py` | Marks the download package, which holds the shared helpers, the provenance rows, the progress files, and the clients and queues for the EPA RSIG web service and the public CMAS bucket on Amazon Web Services. |
| `choi_dl/common.py` | Helpers of the download code: the working folders and the record folder of logs, provenance and configuration, paths relative to the root folder, masking of text written to logs, paired UTC and local time stamps, line wrapping, and the temporary folder of the running process. |
| `choi_dl/dl_rsig.py` | Runs the queue of EPA RSIG requests in a fixed order: the CMAQ NO2 target and the meteorological series month by month for 2015 to 2019, the lag requests for 31 December 2014 and a second daily NO2 series for 2016. It skips items whose saved answer verifies, records and limits refusals, stops when disk runs short, and copies the provenance records. |
| `choi_dl/dl_s3.py` | Runs the S3 queue against the public CMAS bucket in order: the 2016 MCIP files, the daily means of the merged gridded NO, NO2, HONO and VOC emissions for 2017 and 2019, the stack groups archives, and the point and marine sectors summed into cells, each written as a compressed NumPy file with its provenance row. |
| `choi_dl/downloader.py` | The detached downloader: runs every remaining RSIG and S3 request in order with its provenance rows, resuming past files whose rows verify, keeps the progress files, holds a lock so that one instance runs, and writes a closing line with each part's counts. |
| `choi_dl/ne.py` | Downloads the Natural Earth files with cartopy's own downloader, at most one request a second, records each answer's status and headers, then hashes every file and writes its provenance row. |
| `choi_dl/progress.py` | Writes the downloader's progress file (one ASCII line per item) and its status file of counts, replaced whole after each item; a write that fails is retried and then held for the next write, so that a busy file never stops the queue. |
| `choi_dl/prov.py` | Writes and reads the provenance rows of each downloaded file or extract (path, source, byte range, retrieval time, HTTP status, size, sha256, version and related fields) as a text record and a JSON-lines record per folder, verifies a file against its row, and copies the records to the record folder. |
| `choi_dl/rsig.py` | Client of the EPA RSIG web service: builds each monthly request, fetches it by paced streaming requests with three timed retries, treats any refusal or answer that holds no data as a stop, decompresses and checks the netCDF answer, and writes the file with its provenance row. |
| `choi_dl/s3io.py` | Client of the public CMAS bucket on Amazon Web Services: anonymous HEAD and ranged GET requests to one host, paced and retried, with refusals never retried; it parses tar contents files, reads members inside the archives, opens HDF5 members, and computes daily grid means and per-stack means. |
| `config/S2A_PLAN_RSIG.json` | The RSIG download plan: the 677 requests in their order, with the coverage names and the sha256 of each coverage's description. |
| `config/S2A_PLAN_S3.json` | The S3 download plan: the names and ETags of the public objects and archive members read, the day sets, and the point and marine sectors kept. |
| `p2_colab_driver.py` | Driver of the arrays stage on Colab: waits for the RSIG and Amazon Web Services stages to finish, restores their outputs from the Drive, runs the stage's 22 steps in order as separate processes (verification, inventory, Natural Earth, grid and mask, input checks, emissions, round trip, splits, arrays, index, statistics, checks and file list), stops at the first failure, and mirrors outputs to the Drive. |
| `s2a/s2a_s3_common.py` | Helpers for the S3 test scripts: the object keys of the merged, point, stack groups and MCIP files, the tar contents files with their provenance rows, JSON reading and writing, a log and a timer. |
| `s2a/s2a_s3_ne.py` | Fetches the Natural Earth files, hashes each and writes its provenance row, checks by name the 48 contiguous states, the District of Columbia and the ten cities, and writes a summary JSON. |
| `s2c/s2c_analyze.py` | Summarizes the verification results (read only): files and bytes by kind, RSIG items tried more than once, failed checks, decompressed-member checks, stack counts and stacks outside the grid, and, on request, the count of extract files whose field is zero everywhere. |
| `s2c/s2c_backup.py` | Copies each named file beside itself as a backup, never overwriting an earlier backup, and checks that the copy is not empty and that its size and sha256 equal the original's. |
| `s2c/s2c_copies.py` | Compares each source folder's provenance records with their copies in the record folder by size and sha256 (read only). |
| `s2c/s2c_filestat.py` | Prints, for each named file, its size and the first 16 hex digits of its sha256 and, for text files, whether it is 7-bit ASCII with LF line ends and its longest line (read only). |
| `s2c/s2c_inv_check.py` | Checks the inventory file (read only): 7-bit ASCII with LF line ends, valid JSON, a unique path, a sha256 and a byte count for every entry, and every input verified; prints its stop section and its counts by kind. |
| `s2c/s2c_inventory.py` | Builds the inventory of the downloaded inputs from the verification and refetch results: drops any point or marine sector not read in full for both years, stops if a required target, input, lag, MCIP or merged-emission file is missing or failed or the disk is short, and writes the inventory JSON. |
| `s2c/s2c_mtimes.py` | Prints the size and last-modified time of the downloader's progress, status and log files (read only). |
| `s2c/s2c_newrows.py` | Prints the provenance rows that the refetch added for the named dates, masked and shortened, with the record's line count and ASCII test (read only). |
| `s2c/s2c_plan_detail.py` | Prints the details of the S3 download plan (read only): the MCIP files, the merged-emission days, the point and marine sectors, members and stack counts, the stack groups entries and the Natural Earth summary. |
| `s2c/s2c_prov_refresh.py` | Refreshes the record folder's copies of each source folder's provenance records, replacing a differing copy only where a backup of it exists, and compares every pair again. |
| `s2c/s2c_reads.py` | Prints the clock, the free disk and the available memory in decimal GB (read only). |
| `s2c/s2c_refetch.py` | Fetches named point and marine items again through the downloader's own S3 route, one item per call, within a two-hour cap kept in a state file and with its own progress and status files; a refusal stops that item and every later one. |
| `s2c/s2c_rows.py` | Summarizes the provenance record of every source folder (read only): line counts, distinct paths, paths with several rows, field sets, note words and a masked sample row, with a check of each text twin. |
| `s2c/s2c_status.py` | Prints the clock, the free disk and the available physical memory in decimal GB (read only). |
| `s2c/s2c_syntax.py` | Compiles each named script without writing bytecode, prints OK or the error, and removes the stage's own bytecode folder if one is present. |
| `s2c/s2c_verify.py` | Builds the expected set of downloaded files from the two download plans through the download code's own functions that list the items, then verifies every file on disk against its provenance row (sha256, size and checks) and writes the results as JSON, changing no data file. |
| `s2c/s2c_version_check.py` | Checks the version field of every provenance row against the expected EQUATES, CMAQ, MCIP and WRF versions and, for RSIG, the coverage name and the sha256 of its description; prints the counts, every differing row and the HTTP status counts (read only). |
| `s2c/s2c_walk.py` | Lists what is on disk under the working folder's raw, extract, work and Natural Earth subfolders (read only): files, bytes and counts by extension per folder, and every partial, failed or temporary file. |
| `s3/__init__.py` | Marks the preprocessing package, which checks the inputs and builds the land mask, the emission fields, the splits, the daily input and target arrays and the scaling statistics. |
| `s3/s3_assemble.py` | Assembles the channel values of one day as 299 x 459 arrays, shared by the round trip and the array build: the RSIG meteorological fields (wind components as the mean of a cell's four corners), the static MCIP fields, vegetation, the emission map of the day's month and weekday, and the previous day's lagged fields. |
| `s3/s3_build.py` | Builds the yearly arrays for 2015 to 2019 month by month: the float16 and float32 input channels and the NO2 target in ppb (day x 299 x 459), with each day's channel minima and maxima and the smallest positive values used for the statistics; it pauses when free memory falls below 1.5 GB. |
| `s3/s3_clean_year.py` | Removes the incomplete arrays of one year of the build, only when that year's build record is absent, and prints each removed file's name and size. |
| `s3/s3_d4.py` | Checks every input before any array is built: the RSIG answers (raw and decompressed sha256), the MCIP files and the emission extracts; it counts units, dimensions, grid attributes, coverage, fill values and NaN on the land cells, values outside the expected ranges and duplicate dates, and stops when fills exceed 0.1 percent of land cell-days. |
| `s3/s3_d4_allcells.py` | Sums the fill and NaN counts on all cells of every RSIG series and year, and of the lag answers, from the input-check summary (read only). |
| `s3/s3_d4_record.py` | Writes the input-check record and its JSON summary from the input-check counts (coverage, duplicates, units, dimensions and grid, fills and NaN, value ranges, the MCIP and emission checks and what was excluded), and prints the verdict. |
| `s3/s3_emis.py` | Forms the daily emission fields of 2017 and 2019 (NO, NO2 and HONO in mol/s from the merged and sector extracts, with weekday-class means for the representative-day sectors and holidays left out; VOC in g/s), writes them as arrays, and builds the monthly weekday emission climatology and the yearly NOx totals. |
| `s3/s3_files.py` | Lists the files the preprocessing wrote, with path, bytes and the first 16 hex digits of the sha256, as record lines. |
| `s3/s3_files_added.py` | Lists further named files with their bytes and the first 16 hex digits of their sha256 as record lines. |
| `s3/s3_finish.py` | Closes the preprocessing log with an end line, checks the log and the input-check record for 7-bit ASCII, LF line ends and line length, and writes the one-line completion marker. |
| `s3/s3_grid_mask.py` | Checks the grid of the RSIG answers against MCIP's GRIDCRO2D (cell centers within 0.0001 degree, the dot grid offset by half a cell) and builds the land mask: the cells whose center lies inside the 48 contiguous states and the District of Columbia (Natural Earth) and that MCIP marks as land. |
| `s3/s3_index.py` | Writes the index files of the preprocessing: each channel's array, index, source, unit, storage type and round-trip error, the channel sets of the models, and every array with its shape, type, size, sha256 and dates, each sha256 taken again from the file. |
| `s3/s3_inspect.py` | Prints the keys of the inventory and of the two download plans with short samples (read only). |
| `s3/s3_mtimes.py` | Prints the name, size and modification time of each file in the code's bytecode folders (read only). |
| `s3/s3_norm.py` | Computes the scaling statistics of each fold and of each two-year emission test on the training days only: each input channel's minimum and maximum, for precipitation and the emission channels the first percentile c of the positive values and the bounds of log10(x + c), and the target's minimum and maximum over the land cells. |
| `s3/s3_privacy.py` | Counts the preprocessing's text files whose text the masking function would change (a home folder, a sign-in name, an at sign); prints counts only. |
| `s3/s3_probe.py` | Prints the variables, dimensions, grid attributes, units, shapes and time-flag ends of one input of each kind (read only). |
| `s3/s3_rec.py` | Appends checked lines (7-bit ASCII, at most 78 characters, no at sign) from a text file written beforehand to the preprocessing log. |
| `s3/s3_report_parts.py` | Writes the log sections of the round trip, the build, the splits, the statistics and the second reading from the stage's outputs, so that each number is copied by code. |
| `s3/s3_roundtrip.py` | Tests each channel's storage type before the arrays are written: on the first day of each month of 2016 it takes every channel through float16 (pressure through its affine map) and moves to float32 any channel with a non-finite value or a 99th-percentile relative error above 0.001. |
| `s3/s3_second.py` | Reads the NO2 target, pressure and a wind component again from the raw answers for the 15th of each month of 2015 to 2019 by a separate path (hashlib, zlib and SciPy's netCDF reader) and compares them with the stored arrays: the target exactly, the others within float16 rounding. |
| `s3/s3_selfcheck_finite.py` | Checks from the per-day minima and maxima that every stored channel of every day is finite and within float16's finite range (read only). |
| `s3/s3_selfcheck_lags.py` | Checks that each day's lagged channels equal the previous day's stored fields across month and year boundaries, and for 1 January 2015 the lag answers of 31 December 2014 (read only). |
| `s3/s3_selfcheck_norm.py` | Recomputes a few scaling statistics of the fold with 2019 held out directly from the stored arrays over its training days and compares them with the fold's statistics file (read only). |
| `s3/s3_show.py` | Prints a JSON file's subtree with values shortened and at signs masked (read only). |
| `s3/s3_splits.py` | Writes the splits: for each held-out year 2015 to 2019 the training days (the other four years less the validation days, every day of each ISO week whose number is divisible by five), the validation days and the held-out days, and the two-year emission tests, testing that the lists are disjoint and complete. |
| `s3/s3_syntax.py` | Parses each module of the preprocessing package without writing bytecode and checks that it is 7-bit ASCII with LF line ends. |
| `s3/s3common.py` | Common parts of the preprocessing: places, the channel table, the inventory, the reader of a raw RSIG answer (read whole, size and sha256 checked, decompressed and opened in memory), the mean of a cell's four corners for dot-grid fields, dates, weekday classes, clock and memory reads, and log helpers. |
| `s3p2/__init__.py` | Marks the package of the second preprocessing pass, which built the arrays that the trainings read. |
| `s3p2/p2_attr_names.py` | Prints the names, and never the values, of the global attributes and variables of the MCIP files and the attribute names of chosen variables of one RSIG answer (read only). |
| `s3p2/p2_blocks_check.py` | Checks that the month-at-once channel sources of the second pass give the same float64 values as the first pass's per-day sources, for three test months (read only). |
| `s3p2/p2_build.py` | Builds the yearly input and target arrays of the second pass for 2015 to 2019 month by month (float16 and float32 channels and the NO2 target in ppb), leaves the emission slots of 2015, 2016 and 2018 as NaN for the later fill, and keeps the per-day extremes and smallest positive values for the statistics. |
| `s3p2/p2_c7.py` | Builds, for every day of 2015 to 2019, the array of the five emission channels taken from the monthly weekday emission climatology of 2017 and 2019, with the per-day extremes and smallest positive values for its statistics. |
| `s3p2/p2_cellsums.py` | The reducer that sums per-stack emissions into grid cells, copied byte for byte from the Drive extraction code, so that both stages share one definition of the point and marine NOx channel. |
| `s3p2/p2_check_change.py` | Prints the differences of the changed first-pass files against their backups, their sha256 and the channel lists that the changed code gives (read only). |
| `s3p2/p2_checks.py` | Runs the second pass's own checks after the build: a second reading of the target and two input channels from the raw answers, the lagged channels, finiteness and the unfilled NaN slots, and the new channels and arrays checked by separate code against their sources. |
| `s3p2/p2_copy_cellsums.py` | Copies the cell-sum reducer from the Drive extraction code into its own module byte for byte, with a header, its imports and the two constants it uses, and prints the hashes of the source and the copy. |
| `s3p2/p2_d4.py` | Checks every input of the second pass before the build, as the first pass did and extended to the dominant land-use field and the point and marine NOx inputs: the RSIG answers against the published 12 km grid, the MCIP files and the emission extracts. |
| `s3p2/p2_d4_record.py` | Writes the input-check record of the second pass and its JSON summary from the check counts, and prints the verdict. |
| `s3p2/p2_disk.py` | Computes the disk space that the second pass still needs from its file counts and types, and stops when the free space is below that need plus 30 GB. |
| `s3p2/p2_emis.py` | Forms the daily emission fields of 2017 and 2019 for the second pass: gridded NO, NO2, HONO and VOC from the merged extracts, and the point and marine NOx channel as the sum of NO, NO2 and HONO over six sectors with weekday-class means for the representative-day sectors; it also builds the monthly weekday emission climatology. |
| `s3p2/p2_emisdays.py` | Gives the float64 emission fields of each day of 2017 and 2019 by the same calls as the emission step, so that the round trip sees each value before the cast; each day's float32 cast is checked against the stored array. |
| `s3p2/p2_files.py` | Lists the files that the second pass wrote, by group, with path, bytes and the first 16 hex digits of the sha256, as record lines. |
| `s3p2/p2_finish.py` | Closes the second pass's log with an end line, checks the log and the input-check record, and writes the one-line completion marker. |
| `s3p2/p2_grid_mask.py` | Checks the grid of every MCIP file and RSIG answer against the published 12 km grid, records the wind attributes and the MCIP and WRF versions, checks the MCIP variable lists, builds the land mask from Natural Earth's states with lakes removed and MCIP's land flag, and finds the cells of the ten cities. |
| `s3p2/p2_hash_itemb.py` | Hashes the first pass's arrays and configuration files by sha256 and size, writes the list, and compares it with an earlier list to show that they stayed unchanged (read only). |
| `s3p2/p2_hashes.py` | Prints the path, size and sha256 of the files named or matched by patterns (read only). |
| `s3p2/p2_index.py` | Writes the index files of the second pass (each channel's array, index, source, unit, storage and round-trip results; every array with its shape, type, size, sha256 and dates, with the emission slots that wait for the fill), each sha256 taken again from the file. |
| `s3p2/p2_ne_fetch.py` | Downloads two Natural Earth archives (the states and provinces with lakes removed, and the populated places) by paced, retried requests, tests each archive, extracts its shapefile parts, compares them with earlier copies, and writes their provenance rows. |
| `s3p2/p2_norm.py` | Computes the scaling statistics of the second pass for the folds whose channels were all built (the two-year emission tests and the climatological-emission folds): channel minima and maxima on the training days, the constant c and log bounds of the log-scaled channels, and the target's range over the land cells. |
| `s3p2/p2_norm_table.py` | Writes record lines that summarize the statistics files: their sizes, their sha256 and a few of their values. |
| `s3p2/p2_p1_check.py` | Checks the point and marine NOx channel on two sector-days: the per-stack sums of the cell-sum reducer against the extract's species grids and against the second pass's own formation, cell by cell, in float32 steps. |
| `s3p2/p2_privacy.py` | Counts the second pass's text files whose text the masking function or the privacy patterns would change; prints counts only. |
| `s3p2/p2_roundtrip.py` | Tests the storage type of every channel on every stored day: takes each channel through its default type and records the 99th-percentile and largest relative errors and the values made zero or subnormal; a float16 channel that fails in any year is stored as float32 for every year. |
| `s3p2/p2_rt_table.py` | Writes record lines of the round-trip results: each channel's default and final storage type and its yearly errors and counts. |
| `s3p2/p2_s3c_list.py` | Writes the list of files that the later stages copy (arrays, statistics, mask, city cells and emission arrays), each with its size and its sha256 taken again from the file and compared with the stage's records. |
| `s3p2/p2_show.py` | Prints a JSON file's subtree with values shortened and at signs masked, or one key of each child (read only). |
| `s3p2/p2_splits.py` | Writes the splits of the second pass by the first pass's rule and code, adds the climatological-emission folds with the same days as the main folds, and compares the day lists with the first pass's splits. |
| `s3p2/p2common.py` | Common parts of the second pass: its folders, the channel table with the point and marine NOx channel after the emission channels, the default storage types, JSON and file helpers, dates, clock, memory and disk reads, and a log. |

### `5_fills_and_statistics`

| Path | What it does |
|---|---|
| `b2c/copy_c7a_year.py` | Copies one year's Drive extracts into the working folder file by file once the year's check has passed, testing each file's size and sha256 against the year's sums and manifest, copying a mismatch once more, and stopping when the free disk is below the need plus 30 GB. |
| `bc/backup_beside.py` | Copies each named file beside itself as a backup, never overwriting an earlier backup, and checks that the copy has the same size and sha256. |
| `config/S2A_PLAN_S3.json` | The S3 download plan: the names and ETags of the public objects and archive members read, the day sets, and the point and marine sectors kept. |
| `no2_screen/code_a/no2_screen_a.py` | One of two independently written versions of a screen of the yearly NO2 target arrays for isolated high cell values (above 30 ppb) that stand out from their 5 x 5 neighborhood and from the same cell on nearby days; it checks its inputs against their hashes and writes the candidates and counts. |
| `no2_screen/code_b/no2_screen_b.py` | The second, independently written version of the same screen of the NO2 target, which classifies the cells one at a time by explicit loops so that its results can be compared with the first version's. |
| `s2a/s2a_s3_pt.py` | Gives the U.S. federal holiday dates (legal and observed) used for the representative-day sectors, and holds the test of the point and marine route: the stack groups files, per-stack daily means read whole and selectively, the row and column check by the grid projection, and the stack-to-cell sums. |
| `s8/s8_cfg.py` | Updates the index files after a year's emission channels are filled: marks the year built, records its round-trip results and the new array's size and sha256, and keeps the earlier entries under new keys, after checking the year's verification results and the configuration backups. |
| `s8/s8_d4_record.py` | Writes the input-check record of one filled year from the JSON results of the fill. |
| `s8/s8_d7.py` | Computes the scaling statistics of the five main folds (2015 to 2019 held out) by a rule fixed in advance, on each fold's training days only, with the filled emission channels for 2015, 2016 and 2018, and writes the five statistics files. |
| `s8/s8_d7_selftest.py` | Tests the day selection of the main folds' statistics on small synthetic arrays with traps (extreme validation days, scaled years, NaN slots), compares each result with a brute-force computation, and checks that every wrong input stops the computation. |
| `s8/s8_run.py` | Fills the emission channels of one year of 2015, 2016 or 2018 in its float32 input array: checks the disk, the extracts and the array before writing, fills each day whole from the extracts, and verifies the result, each step writing its JSON result. |
| `s8/s8common.py` | Common parts of the fill: places, the year's extracts with their checks, holidays, the daily emission fields formed as for 2017 and 2019 (gridded NO, NO2, HONO and VOC; the point and marine NOx sum over six sectors), the channel indices, and memory and disk reads. |
| `s8_colab_driver.py` | Driver of the fill and statistics stage on Colab: waits for the arrays stage and the year checks, restores files from the Drive, copies and fills the emission channels of 2018, 2015 and 2016 with their checks and uploads, runs the NO2 screen only when needed, then computes the main folds' statistics and writes the end marker. |

### `6_shared`

| Path | What it does |
|---|---|
| `baselines/__init__.py` | Marks the baselines package: the common fold handling and prediction writer, the climatologies, the weather regression, the gradient-boosted trees with their linear model, and the held-out metrics. |
| `baselines/clim.py` | CLIM-C (each cell's mean target over the training days) and CLIM-WD, the weekday climatology: a day-of-year mean with a 61-day triangular weight times a per-cell weekday factor, every sum in float64, clipped at 0. |
| `baselines/common.py` | Common parts of the baselines: a fold's training and held-out days and read-only arrays, the land cells, calendar position, weekday and season, the prediction writer with its done marker, and a free-memory floor of 1.5 GB. |
| `baselines/gbt.py` | The gradient-boosted trees (scikit-learn's histogram gradient boosting) and the linear model on the same features: the target minus the weekday climatology, from the inputs at the cell, their 3 x 3, 9 x 9 and 27 x 27 means, upwind NOx, calendar features and the climatology. |
| `baselines/metreg.py` | MET-REG, the weather regression: for each cell a ridge regression of ln(y + 0.1) - ln(C + 0.1) on 14 standardized meteorological predictors, with C the weekday climatology, returned to ppb with a smearing factor and clipped at 0. |
| `baselines/run.py` | Entry of the baselines: fits each named baseline on each fold's training days, predicts the held-out year, writes the predictions with done markers and the held-out metrics to sealed files, and skips any baseline and fold already done. |
| `baselines/sealed.py` | Writes the baselines' held-out metrics (n, R, IOA, MAE, mean error and RMSE per year and season, and the series of the ten cities for 2016) to sealed files, and returns only paths, sizes and hashes. |
| `choi/__init__.py` | Marks the core package: the networks, the IOA loss, the metrics, the whole-grid inference, the tar-header reader, the float16 storage and the stack-to-cell sums; importing it loads no heavy package. |
| `choi/f16.py` | Float16 storage of the arrays: maps pressure from Pa to [0, 1] by (x - 50,000) / 60,000 before the cast and back, casts other channels plainly, and tests a channel's round trip, keeping it in float32 when it has non-finite values or a 99th-percentile relative error above 0.001. |
| `choi/infer.py` | Predicts the whole 299 x 459 grid from 64 x 64 windows at stride 32, blended by a two-dimensional Hann weight with a floor of 0.001 on an edge-padded grid, and returns the field in ppb with negative values set to zero and counted. |
| `choi/loss.py` | The training loss, 1 - IOA over the masked cells of a batch, computed as the ratio of the squared error to the potential error, with a guard that gives zero for a zero denominator. |
| `choi/metrics.py` | The evaluation metrics R, IOA, MAE, mean error, MSE, RMSE, NMB, NME and R2 over paired emulated and CMAQ values, and the skill score against a reference, every sum in float64 with an optional mask; a non-finite pair raises an error. |
| `choi/nets.py` | The networks in PyTorch: a U-Net close to the published design, a U-Net with residual blocks, squeeze-and-excitation gates and dropout, and a per-pixel network, with helpers for widths, dropout and counts of parameters and multiply-accumulate operations. |
| `choi/stacks.py` | Sums point and marine emission stacks into grid cells: takes each stack's daily mean of NO, NO2 and HONO over the 24 hourly records and adds it to its cell of the 299 x 459 grid by the row and column in the stack groups file, counting the stacks outside the grid. |
| `choi/tario.py` | Reads tar headers by byte range without unpacking: member names (ustar, GNU long names and pax records), sizes, times, types and offsets, each header checksum verified, from any function that returns a byte range. |
| `choi_dl/__init__.py` | Marks the download package, which holds the shared helpers, the provenance rows, the progress files, and the clients and queues for the EPA RSIG web service and the public CMAS bucket on Amazon Web Services. |
| `choi_dl/common.py` | Helpers of the download code: the working folders and the record folder of logs, provenance and configuration, paths relative to the root folder, masking of text written to logs, paired UTC and local time stamps, line wrapping, and the temporary folder of the running process. |
| `choi_dl/dl_rsig.py` | Runs the queue of EPA RSIG requests in a fixed order: the CMAQ NO2 target and the meteorological series month by month for 2015 to 2019, the lag requests for 31 December 2014 and a second daily NO2 series for 2016. It skips items whose saved answer verifies, records and limits refusals, stops when disk runs short, and copies the provenance records. |
| `choi_dl/dl_s3.py` | Runs the S3 queue against the public CMAS bucket in order: the 2016 MCIP files, the daily means of the merged gridded NO, NO2, HONO and VOC emissions for 2017 and 2019, the stack groups archives, and the point and marine sectors summed into cells, each written as a compressed NumPy file with its provenance row. |
| `choi_dl/downloader.py` | The detached downloader: runs every remaining RSIG and S3 request in order with its provenance rows, resuming past files whose rows verify, keeps the progress files, holds a lock so that one instance runs, and writes a closing line with each part's counts. |
| `choi_dl/ne.py` | Downloads the Natural Earth files with cartopy's own downloader, at most one request a second, records each answer's status and headers, then hashes every file and writes its provenance row. |
| `choi_dl/progress.py` | Writes the downloader's progress file (one ASCII line per item) and its status file of counts, replaced whole after each item; a write that fails is retried and then held for the next write, so that a busy file never stops the queue. |
| `choi_dl/prov.py` | Writes and reads the provenance rows of each downloaded file or extract (path, source, byte range, retrieval time, HTTP status, size, sha256, version and related fields) as a text record and a JSON-lines record per folder, verifies a file against its row, and copies the records to the record folder. |
| `choi_dl/rsig.py` | Client of the EPA RSIG web service: builds each monthly request, fetches it by paced streaming requests with three timed retries, treats any refusal or answer that holds no data as a stop, decompresses and checks the netCDF answer, and writes the file with its provenance row. |
| `choi_dl/s3io.py` | Client of the public CMAS bucket on Amazon Web Services: anonymous HEAD and ranged GET requests to one host, paced and retried, with refusals never retried; it parses tar contents files, reads members inside the archives, opens HDF5 members, and computes daily grid means and per-stack means. |
| `config/TIER_A_CONFIG.json` | The configuration of the training, baseline and evaluation runs: the places on the Drive, the models, the compute-unit rates and limits, and the rules that the drivers test before a run. |
| `session_probe.py` | Writes the runtime's environment (Python, PyTorch, CUDA and cuDNN versions, NumPy, the GPU and its memory) to a JSON file, and stops when no CUDA device is visible. |
| `stage9_colab_driver.py` | Driver that fits the gradient-boosted trees and their linear model on a CPU runtime through the trainer shim, reading the same arrays and configuration files as the trainings and writing sealed predictions and metrics. |
| `tier_a_rules.py` | Reads the rules of the training and evaluation runs from the one configuration file: the reasons to refuse a run, the departures it records, the choice of runtime, the compute-unit rates, the projected hours and the units spent; standard library only. |
| `train/__init__.py` | Marks the training package: the shared network and training-step module, the configuration and plan, the data reader, the training loop, the inference, the timing and equivalence tests and the trainer. |
| `train/bench.py` | Times training steps of one network, width and device: warm-up and timed steps on synthetic tensors and on real batches, the median seconds per step, and the peak memory, written as JSON; no loss or weight is kept. |
| `train/cfg.py` | Paths, the design's fixed values and the configuration files read by name (arrays, channels, splits, statistics and the runtime plan), with checks of the fields that the trainer expects. |
| `train/data.py` | The data reader of the trainer and the inference: reads the arrays memory-mapped, joins the float16 and float32 channels, applies the log and min-max scaling from the statistics files, draws the training tiles and the fixed validation tiles, and holds the splits with their guards. |
| `train/emul_core.py` | The shared module of the networks and the training step: a U-Net close to the published design, a U-Net with residual blocks and squeeze-and-excitation gates, and a per-pixel network; the 1 - IOA loss with Adam and its learning-rate decay; deterministic flags and their checks; device-independent random streams; environment reports and checkpoint helpers. |
| `train/equiv.py` | Runs a 200-step synthetic training with one seed on a given device, from identical starting weights and batches, and writes every step's loss, so that devices can be compared. |
| `train/equiv_test.py` | The equivalence test of the training code: one forward and backward pass on fixed tensors and weights and a 200-step synthetic training, for each network on one device, compared with the CPU where the device differs. |
| `train/f19.py` | An exploratory emission-scaling check of trained networks: predicts the held-out year again with the NO, NO2, HONO and point and marine NOx inputs scaled by 0.9, and computes at 1,000 fixed cells the derivative of the emulated NO2 with respect to those inputs. |
| `train/flags.py` | Sets the device, precision flags, threads, seeds and random states of the trainer and the test entries from the shared module, and reports them in every training log. |
| `train/loop.py` | The loop of one training: the 1 - IOA loss with Adam and step decay, batches of 64 tiles, validation every 250 steps on 1,024 fixed tiles with early stopping, the best state saved and reloaded for prediction, resumable checkpoints, and stops on low memory or non-finite values. |
| `train/plan.py` | The list of trainings in their order, with model ids, seeds, labels and network settings, the checks of the runtime plan, and the projected run time of each training. |
| `train/predict.py` | Predicts a training's day sets with the best saved network by whole-grid inference, writes the land-cell predictions in ppb (and the full grid for 2016) with an information file, and writes and checks the training's done marker. |
| `train/progress.py` | Writes the trainer's progress file and status file, each write retried and never raising an error. |
| `train/tinyfit.py` | A short fit before the main trainings: each network trains 300 steps on 8 fixed tiles, and the loss at steps 0, 100, 200 and 300 is recorded as a check that the code learns. |
| `train/trainer_main.py` | The detached trainer: for each training of the runtime plan, skips it when its done marker verifies, else reads its data, trains it with resumable checkpoints, predicts the held-out year and writes the done marker, within a time budget and with one running instance. |
| `train/vendor/compare_core.py` | Compares the device test and the synthetic training between two devices: relative differences of outputs, losses and gradients against fixed tolerances (0.001 for the device test, 2 percent for the final training loss). |
| `train_colab_driver.py` | Driver of the trainings on a Colab GPU runtime: checks the earlier stages' markers, copies the configuration files and arrays with size checks, runs the equivalence test, the timing and the trainings through the trainer shim, records each runtime's use, and mirrors outputs to the Drive; the later drivers import its helpers. |
| `train_shim.py` | Runs a trainer script as the main program after setting the overrides of the training package's constants that the configuration file lists; with an empty list the script runs unchanged. |

### `6_cpu_baselines`

| Path | What it does |
|---|---|
| `baselines/clim_em.py` | CLIM-EM, the emission-scaled climatology: the weekday climatology times the ratio of the day's mean NOx emission over the 9 x 9 cells around each cell to its climatological value fitted on the training days. |
| `baselines/gbt_curve.py` | Records the validation RMSE of the gradient-boosted trees every 25 rounds on the validation days with the round of its minimum, and fits GBT-ES (stopped at that round) for the evaluation's printing rule; the curve goes to a sealed file only. |
| `baselines/p7base.py` | Shared parts of the refitted baselines: a fold that guards every read by the day's role, the cross-fitted weekday climatology read from its tables, and the writer of the held-back-day predictions with their done markers. |
| `baselines/run_p7.py` | Refits the baselines CLIM-C, CLIM-WD, MET-REG, GBT, GBT-RIDGE, CLIM-EM and GBT-ES (the trees stopped at their best validation round) on each fold's training days with the cross-fitted weekday climatology, and writes their predictions for the test days and the held-back days to sealed files. |
| `config/TIER_A_CONFIG.json` | The configuration of the training, baseline and evaluation runs: the places on the Drive, the models, the compute-unit rates and limits, and the rules that the drivers test before a run. |
| `cpu/__init__.py` | Marks the package of the CPU steps: the read-only pass over the arrays, the statistics rewritten without the held-back days, and the climatology tables. |
| `cpu/ctables.py` | Builds the climatology tables of each fold: the weekday climatology fitted on all training years and on each set of three, so that each training year gets values fitted without it (cross-fitted), and the tables of the two-year emission tests. |
| `cpu/norm_f6.py` | Rewrites the scaling statistics of the climatological-emission folds 2016 and 2018 and of the two-year emission tests with the held-back and buffer days left out of the training days, by a rule fixed in advance. |
| `cpu/pd17_pass.py` | A read-only pass over every built input array before any fit: checks its sha256, counts non-finite, fill and negative values per channel (stopping on non-finite or fill values), and tests the four-corner mean of the dot-grid winds on a synthetic case. |
| `cpu_colab_driver.py` | Driver of the CPU baselines stage on Colab: checks the earlier stages' markers and the runtime, copies the configuration files and arrays with size checks, then runs the read-only pass, the statistics rewrite, the climatology tables and the baselines, mirroring results to the Drive. |
| `s3/__init__.py` | Marks the preprocessing package, which checks the inputs and builds the land mask, the emission fields, the splits, the daily input and target arrays and the scaling statistics. |
| `s3/s3_assemble.py` | Assembles the channel values of one day as 299 x 459 arrays, shared by the round trip and the array build: the RSIG meteorological fields (wind components as the mean of a cell's four corners), the static MCIP fields, vegetation, the emission map of the day's month and weekday, and the previous day's lagged fields. |
| `s3/s3_build.py` | Builds the yearly arrays for 2015 to 2019 month by month: the float16 and float32 input channels and the NO2 target in ppb (day x 299 x 459), with each day's channel minima and maxima and the smallest positive values used for the statistics; it pauses when free memory falls below 1.5 GB. |
| `s3/s3_norm.py` | Computes the scaling statistics of each fold and of each two-year emission test on the training days only: each input channel's minimum and maximum, for precipitation and the emission channels the first percentile c of the positive values and the bounds of log10(x + c), and the target's minimum and maximum over the land cells. |
| `s3/s3common.py` | Common parts of the preprocessing: places, the channel table, the inventory, the reader of a raw RSIG answer (read whole, size and sha256 checked, decompressed and opened in memory), the mean of a cell's four corners for dot-grid fields, dates, weekday classes, clock and memory reads, and log helpers. |
| `s3p2/__init__.py` | Marks the package of the second preprocessing pass, which built the arrays that the trainings read. |
| `s3p2/p2_norm.py` | Computes the scaling statistics of the second pass for the folds whose channels were all built (the two-year emission tests and the climatological-emission folds): channel minima and maxima on the training days, the constant c and log bounds of the log-scaled channels, and the target's range over the land cells. |
| `s3p2/p2common.py` | Common parts of the second pass: its folders, the channel table with the point and marine NOx channel after the emission channels, the default storage types, JSON and file helpers, dates, clock, memory and disk reads, and a log. |
| `train/guard.py` | The roles guard: gives each day of a fold its role (training, validation, test, held-back or buffer) from the splits and the held-back days, and refuses any read of a day for a purpose that its role does not allow. |

### `6_cpu_norm_2019`

| Path | What it does |
|---|---|
| `config/TIER_A_CONFIG.json` | The configuration of the training, baseline and evaluation runs: the places on the Drive, the models, the compute-unit rates and limits, and the rules that the drivers test before a run. |
| `cpu/__init__.py` | Marks the package of the CPU steps: the read-only pass over the arrays, the statistics rewritten without the held-back days, and the climatology tables. |
| `cpu/norm_f6a_2019.py` | Rewrites the scaling statistics of the climatological-emission fold with 2019 held out, with the held-back and buffer days left out of the training days, by the same code as the earlier rewrite. |
| `cpu_t2_driver.py` | Driver on Colab that rewrites the statistics of the climatological-emission fold with 2019 held out: checks that the file is not yet written and that the CPU baselines stage has finished, copies the inputs, runs the rewrite, and puts the new file and its index on the Drive. |
| `s3/__init__.py` | Marks the preprocessing package, which checks the inputs and builds the land mask, the emission fields, the splits, the daily input and target arrays and the scaling statistics. |
| `s3/s3_assemble.py` | Assembles the channel values of one day as 299 x 459 arrays, shared by the round trip and the array build: the RSIG meteorological fields (wind components as the mean of a cell's four corners), the static MCIP fields, vegetation, the emission map of the day's month and weekday, and the previous day's lagged fields. |
| `s3/s3_build.py` | Builds the yearly arrays for 2015 to 2019 month by month: the float16 and float32 input channels and the NO2 target in ppb (day x 299 x 459), with each day's channel minima and maxima and the smallest positive values used for the statistics; it pauses when free memory falls below 1.5 GB. |
| `s3/s3_norm.py` | Computes the scaling statistics of each fold and of each two-year emission test on the training days only: each input channel's minimum and maximum, for precipitation and the emission channels the first percentile c of the positive values and the bounds of log10(x + c), and the target's minimum and maximum over the land cells. |
| `s3/s3common.py` | Common parts of the preprocessing: places, the channel table, the inventory, the reader of a raw RSIG answer (read whole, size and sha256 checked, decompressed and opened in memory), the mean of a cell's four corners for dot-grid fields, dates, weekday classes, clock and memory reads, and log helpers. |
| `s3p2/__init__.py` | Marks the package of the second preprocessing pass, which built the arrays that the trainings read. |
| `s3p2/p2_norm.py` | Computes the scaling statistics of the second pass for the folds whose channels were all built (the two-year emission tests and the climatological-emission folds): channel minima and maxima on the training days, the constant c and log bounds of the log-scaled channels, and the target's range over the land cells. |
| `s3p2/p2common.py` | Common parts of the second pass: its folders, the channel table with the point and marine NOx channel after the emission channels, the default storage types, JSON and file helpers, dates, clock, memory and disk reads, and a log. |
| `t2_guard.py` | Copies of the spend guard's functions, which read each runtime's compute-unit rate and stop the stage when its recorded spend would pass the limit. |
| `train/guard.py` | The roles guard: gives each day of a fold its role (training, validation, test, held-back or buffer) from the splits and the held-back days, and refuses any read of a day for a purpose that its role does not allow. |

### `7_train_reference`

| Path | What it does |
|---|---|
| `tier_a_rules.py` | The rules module of the training runs, extended so that the GPU type is the one Colab assigns, kept when its own checks pass: the order of types, the reasons to refuse a type, and the choice for the connected type. |
| `train_colab_driver.py` | Driver of the reference U-Net trainings on a Colab GPU runtime: the shared training driver with the GPU type taken as Colab assigns it when its own checks pass. |

### `7_train_residual`

| Path | What it does |
|---|---|
| `config/TIER_A_CONFIG.json` | The configuration of the training, baseline and evaluation runs: the places on the Drive, the models, the compute-unit rates and limits, and the rules that the drivers test before a run. |
| `tier_a_rules.py` | The rules module of the residual trainings: the shared rules plus the GPU type taken as Colab assigns it, the two lanes, the checks of the design's training rows and step counts, and the guard that stops a run when its recorded compute-unit spend passes the limit. |
| `train/cfg.py` | Paths, fixed values and configuration files of the trainer, extended with the learning-rate period and patience of each training, the climatology tables and the albedo channel of the 18-input network. |
| `train/data.py` | The data reader of the residual trainings: the shared reader with the held-back days removed from every split, the cross-fitted weekday climatology read from its tables, and the residual target (the target minus the climatology, scaled) for the residual networks. |
| `train/emul_core.py` | The shared network and training-step module with a mean squared error loss beside 1 - IOA, gradient-norm clipping, a final layer initialized at zero for the residual networks, and the per-cell network PIX-R. |
| `train/guard.py` | The roles guard: gives each day of a fold its role (training, validation, test, held-back or buffer) from the splits and the held-back days, and refuses any read of a day for a purpose that its role does not allow. |
| `train/loop.py` | The training loop with the loss chosen per training (1 - IOA or mean squared error), the learning-rate period and patience from the plan, an optional gradient clip, the gradient norm logged before each validation, and one profiling window. |
| `train/plan.py` | The list of trainings: the shared plan plus the design's rows (the residual U-Net, the steps from the reference U-Net, the climatological-emission and 18-input variants, the per-cell network and the extra seeds) with their seeds and settings. |
| `train/predict.py` | The inference of the residual trainings: the shared inference with the field returned as the climatology plus the rescaled prediction, clipped at 0 ppb, a check that the blend covers every land cell, and the held-back-day predictions written to sealed files. |
| `train/trainer_main.py` | The detached trainer of the design's trainings: runs its rows in two lanes with a budget stop, the climatology tables checked first, and, on request, the held-back-day predictions of every finished training. |
| `train2_colab_driver.py` | Driver of the residual and other design trainings on Colab A100 runtimes in two lanes: loads the shared package, overlays this folder's files, runs each lane's trainings through the trainer, writes the held-back-day predictions, records each runtime's spend, and mirrors outputs to the Drive. |

### `8_evaluation`

| Path | What it does |
|---|---|
| `config/TIER_A_CONFIG.json` | The configuration of the training, baseline and evaluation runs: the places on the Drive, the models, the compute-unit rates and limits, and the rules that the drivers test before a run. |
| `eval/__init__.py` | Marks the evaluation package: the readers and writer of the sealed files, the metrics and strata, the four comparisons fixed in advance, the exploratory analyses AN-02 and AN-05, the labels, the printing rule of the early-stopped trees (GBT-ES) and the entry. |
| `eval/an02.py` | AN-02: for each training and fold, the per-cell range of each emission input over the training days, and the share of test cell-days with an emission input outside its cell's range, by held-out year and urban class. |
| `eval/an05.py` | AN-05: on the 2016 and 2018 folds, the residual U-Net (M) against the same network with climatological emissions (M-C7) and their difference, each with the AN-02 share, the F-12 weekday slope (the slope of the model's Tuesday-to-Friday minus Sunday map on CMAQ's) and its count of cell-days. |
| `eval/d8in.py` | Finds, for each model and fold, the test days with any input outside its training range (the per-channel minimum and maximum over all cells of the training days), for the error on those days. |
| `eval/family.py` | The four comparisons fixed in advance, C-1 to C-4: a seeded paired moving-block bootstrap of whole test days within each year, statistics from per-day sums, percentile intervals, Holm's step-down procedure with the margins, the seed rule, the 14-day sensitivity, the exploratory intervals and the held-back-day replication. |
| `eval/gbt_es.py` | Applies the printing rule of GBT-ES, the gradient-boosted trees stopped at the round of least validation RMSE: it is reported beside the trees only when the validation RMSE at round 500 is more than 1 percent above the curve's minimum. |
| `eval/io.py` | The readers of the sealed predictions of the networks and baselines, each read only after its done marker and sha256 verify; the target read through the roles guard; the held-back days read only under an opening record; and the sealed writer of JSON, CSV and done files with sha256. |
| `eval/labels.py` | The labels and notes that word each F-12 weekday-contrast row by model; they fix the wording of the rows only. |
| `eval/metrics.py` | The metrics, strata and diagnostics of the evaluation (n, MB, MAE, RMSE, NMB, NME, R, R2, IOA, the skill score, the anomaly correlation, the map RMSE, deciles, exceedance scores, the spread ratio, the spatial spectrum, the weekday contrast and others), every sum in float64; its docstring defines each one. |
| `eval/run_eval.py` | The entry of the evaluation: per fold it reads the target, the classes and each model's sealed predictions and computes the metrics by stratum, the diagnostics, the per-day sums for the comparisons and the figures' data; then the comparisons, the exploratory analyses and the five-fold summaries, all written to a new sealed folder. |
| `eval_colab_driver.py` | Driver of the evaluation on a high-memory Colab CPU runtime: checks the package files against their sums, copies the configuration files and arrays, evaluates the test days of the five folds and then, behind its gate, the held-back days, and records the compute units used. |
| `nb9_gate.py` | The gates of the evaluation: the test-day evaluation runs only when every listed training and baseline has a verified done marker and the opening record is present, and the held-back-day evaluation also needs its own opening record; sealed files are tested by name and size only. |
| `nb9_ledger.py` | The spend ledger and guard of the evaluation runtime: sums the recorded compute units of every runtime record at the configured rates and stops when the total passes the limit. |
| `pe/code/eval/__init__.py` | Marks the evaluation package: the readers and writer of the sealed files, the metrics and strata, the four comparisons fixed in advance, the exploratory analyses AN-02 and AN-05, the labels, the printing rule of the early-stopped trees (GBT-ES) and the entry. |
| `pe/code/eval/an02.py` | AN-02: for each training and fold, the per-cell range of each emission input over the training days, and the share of test cell-days with an emission input outside its cell's range, by held-out year and urban class. |
| `pe/code/eval/an05.py` | AN-05: on the 2016 and 2018 folds, the residual U-Net (M) against the same network with climatological emissions (M-C7) and their difference, each with the AN-02 share, the F-12 weekday slope (the slope of the model's Tuesday-to-Friday minus Sunday map on CMAQ's) and its count of cell-days. |
| `pe/code/eval/d8in.py` | Finds, for each model and fold, the test days with any input outside its training range (the per-channel minimum and maximum over all cells of the training days), for the error on those days. |
| `pe/code/eval/family.py` | The four comparisons fixed in advance, C-1 to C-4: a seeded paired moving-block bootstrap of whole test days within each year, statistics from per-day sums, percentile intervals, Holm's step-down procedure with the margins, the seed rule, the 14-day sensitivity, the exploratory intervals and the held-back-day replication. |
| `pe/code/eval/gbt_es.py` | Applies the printing rule of GBT-ES, the gradient-boosted trees stopped at the round of least validation RMSE: it is reported beside the trees only when the validation RMSE at round 500 is more than 1 percent above the curve's minimum. |
| `pe/code/eval/io.py` | The readers of the sealed predictions of the networks and baselines, each read only after its done marker and sha256 verify; the target read through the roles guard; the held-back days read only under an opening record; and the sealed writer of JSON, CSV and done files with sha256. |
| `pe/code/eval/labels.py` | The labels and notes that word each F-12 weekday-contrast row by model; they fix the wording of the rows only. |
| `pe/code/eval/metrics.py` | The metrics, strata and diagnostics of the evaluation (n, MB, MAE, RMSE, NMB, NME, R, R2, IOA, the skill score, the anomaly correlation, the map RMSE, deciles, exceedance scores, the spread ratio, the spatial spectrum, the weekday contrast and others), every sum in float64; its docstring defines each one. |
| `pe/code/eval/run_eval.py` | The entry of the evaluation: per fold it reads the target, the classes and each model's sealed predictions and computes the metrics by stratum, the diagnostics, the per-day sums for the comparisons and the figures' data; then the comparisons, the exploratory analyses and the five-fold summaries, all written to a new sealed folder. |
