# Results

The outputs of the report's two evaluations, the grid and split files they read, and the published values the report sets beside its own. Every file is listed with its sha256 in `SHA256SUMS.txt`. Text in the JSON and CSV files (notes, descriptions and labels) was edited for publication; no number was changed. The model and analysis labels used in the files (M, P, GBT, CLIM-WD, C-1 to C-4 and the others) are defined in `plan/ANALYSIS_PLAN.md`. Units: concentrations, RMSE, MB (mean bias, the model minus CMAQ; the report's mean error, ME), MAE and the differences C-2 to C-4 are in ppb of surface NO<sub>2</sub>; skill scores, correlations, R2, IOA, NMB, NME, shares and counts have no unit.

## evaluation_test_years/ and evaluation_held_back/

`evaluation_test_years/` holds the first evaluation, on the test days of each held-out year (leave one year out, 2015 to 2019); `evaluation_held_back/` the second evaluation, which repeated it and also evaluated the held-back days once. The two hold the same kinds of files. The tables also come as CSV files of the same name (eval_run.json and EVAL_DONE.json excepted); a CSV may hold its JSON's rows in a flatter form, so the counts of rows can differ (d11_curves, diagnostics, family).

| File | What it holds |
|---|---|
| `metrics.json` | Agreement with CMAQ of every model, seed and held-out year (fold), over all land cells and by season, urban class and coast or edge class: RMSE, MB, MAE, NMB, NME, R, R2, r_squared, IOA, SS, with the counts of cell-days and cells (R2 is one minus the squared error over the variance about CMAQ's mean, r_squared the squared correlation) |
| `five_fold.json` | The means of the five yearly values of each model with five held-out years, and the year-shift rows |
| `family.json` | The four comparisons C-1 to C-4: points, Holm levels and intervals, the 14-day sensitivity, the seed rule, 2019 alone and (second evaluation) the held-back days |
| `diagnostics.json` | Per model, seed and fold: map RMSE, spatial and temporal correlations, the anomaly correlation, cell shares, the near-zero, out-of-range, clip, under-training and inference-time diagnostics |
| `deciles.json` | Errors by decile of CMAQ's values |
| `exceedance.json` | Exceedance skill (POD, FAR, CSI, ETS) at three levels |
| `band_ratios.json` | Spatial power of each model's field over CMAQ's, by wavelength band |
| `residual_acf.json` | Autocorrelation of each cell's daily error by lag and season |
| `d4_to_d7.json` | The diagnostics by season and urban class, by coast and edge class, by boundary-layer height decile, and the seasonal residuals |
| `exploratory.json`, `exploratory_strata.json` | The exploratory comparisons and their intervals |
| `f12.json` | The weekday-weekend response of each model against CMAQ's |
| `an02.json`, `an05.json` | The share of test cell-days with an emission input outside the cell's training range, and the comparison of M with M-C7 |
| `gbt_es.json` | The gradient-boosted trees' validation check of their number of rounds |
| `city_series.json` | Agreement in the ten cities, 2016 |
| `d11_curves.json` | The validation curves of the trainings |
| `eval_run.json` | What the evaluation read and ran: models, settings, resampling seed and counts |
| `EVAL_DONE.json` | The sha256 and size of every file the evaluation wrote, as deposited |

The NumPy arrays (`.npy`) are named by kind, held-out year (fold), model and seed; each row of 53,594 values follows the land cells of `grid/mask_conus_land.npy` in the array's storage order.

| Array | What it holds |
|---|---|
| `F3_hist_fold<year>_<model>.npy` | Counts of cell-days on a 200 by 200 grid of value bins, model against CMAQ (Figure 1) |
| `F4_mean_fold<year>_<model>.npy` | Test-day mean maps of each land cell, three rows (Figures 2 and A3) |
| `F5_city_fold2016_<model>_<city>.npy` | Daily series of CMAQ and the model in one city, 2016 test days (Figures 3 and 4) |
| `F5_domain_fold<year>_<model>.npy` | Daily means over the land cells of CMAQ and the model (Figure A5) |
| `D7_maps_fold<year>_<model>_s<seed>.npy` | Seasonal mean differences between the model and CMAQ for each land cell, four rows (winter, spring, summer, fall) |
| `D2_nearzero_fold<year>_<model>_s<seed>.npy` | Counts per land cell of the near-zero diagnostic |
| `an02_fold<year>_<set>_<channels>_min.npy`, `..._max.npy` | Each land cell's lowest and highest training value of each of the five emission channels |

## grid/

| File | What it holds |
|---|---|
| `mask_conus_land.npy` | The 53,594 land cells of the 299 by 459 grid (boolean) |
| `LWMASK.npy` | MCIP's land-water mask as the maps use it |
| `CITY_CELLS.json` | The ten cities and their cells, by Natural Earth's populated places |
| `SPLITS.json` | The training, validation and test days of each fold |
| `LOCKBOX_DAYS.json` | The held-back days and their buffer days |
| `testdays_cv.json` | The test days of each held-out year |
| `MODELS_TIER_A.json` | The models table the evaluation read (86 rows: 51 networks and 35 baselines) |

## report/

`numbers.json` holds every number the report prints, by key, with its printed text and its value; `numbers.tex` holds the same as LaTeX definitions. The figure scripts read them to check that each value they draw or write in a caption equals the report's.

## published/

`PUBLISHED_VALUES.txt` holds the values printed by Payami et al. (2025) for their NO<sub>2</sub> emulator, with their pages, as the table code reads them.

## The data use policy

`EQUATES_Data_Use_Policy.pdf` is EPA's EQUATES Data Use Policy Statement, which asks that a copy of the policy accompany the data passed on, and that published material cite the datasets (DOI 10.15139/S3/F2KJSK) and the emissions data (Foley et al., 2023); they are cited here as follows.

- U.S. Environmental Protection Agency (U.S. EPA) (2021). EQUATES CMAQv5.3.2 output (version 1.0) [data file and documentation]. <https://doi.org/10.15139/S3/F2KJSK>. Accessed 24 October 2026 through EPA's Remote Sensing Information Gateway and the CMAS Data Warehouse (Amazon Web Services and Google Drive).
- Foley, K. M., Pouliot, G. A., Eyth, A., Aldridge, M. F., Allen, C., Appel, K. W., Bash, J. O., Beardsley, M., Beidler, J., Choi, D., Farkas, C., Gilliam, R. C., Godfrey, J., Henderson, B. H., Hogrefe, C., Koplitz, S. N., Mason, R., Mathur, R., Misenis, C., Possiel, N., Pye, H. O. T., Reynolds, L., Roark, M., Roberts, S., Schwede, D. B., Seltzer, K. M., Sonntag, D., Talgo, K., Toro, C., Vukovich, J., Xing, J., Adams, E. (2023). 2002&ndash;2017 anthropogenic emissions data for air quality modeling over the United States. *Data in Brief* 47, 109022. <https://doi.org/10.1016/j.dib.2023.109022>
