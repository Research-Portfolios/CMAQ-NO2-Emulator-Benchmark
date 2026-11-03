# Figure and table code

This folder holds the code that draws the report's figures and writes its tables from the deposited outputs in `results/`. Run each script from the repository root, for example `python report/build_fig_F1.py`; a figure script writes its figure (PNG and PDF), caption, alternative text and plotted values into `build/figures/`, and `report/build_tables.py` writes the tables into `build/tex/`. The maps need the NHGIS state boundaries described in the repository's `README.md` (Reproducing).

## Scripts

Each display of the report, its file in `figures/` (for a figure), the script that makes it and the script's output.

| Display in the report | File in `figures/` | Script | Output |
|---|---|---|---|
| Figure 1 | `figures/fig_F1.png` | `report/build_fig_F1.py` | `build/figures/fig_F1.png` |
| Figure 2 | `figures/fig_F2.png` | `report/build_fig_maps.py` | `build/figures/fig_F2.png` |
| Figure 3 | `figures/fig_F3.png` | `report/build_fig_F3.py` | `build/figures/fig_F3.png` |
| Figure 4 | `figures/fig_F3k.png` | `report/build_fig_F3k.py` | `build/figures/fig_F3k.png` |
| Figure 5 | `figures/fig_F4.png` | `report/build_fig_F4.py` | `build/figures/fig_F4.png` |
| Figure 6 | `figures/fig_F5.png` | `report/build_fig_F5.py` | `build/figures/fig_F5.png` |
| Figure 7 | `figures/fig_F6.png` | `report/build_fig_F6.py` | `build/figures/fig_F6.png` |
| Figure 8 | `figures/fig_A_F6.png` | `report/build_fig_A_F6.py` | `build/figures/fig_A_F6.png` |
| Figure 9 | `figures/fig_F7.png` | `report/build_fig_maps.py` | `build/figures/fig_F7.png` |
| Figure A1 | `figures/fig_ad1_design.png` | `report/build_fig_ad1_design.py` | `build/figures/fig_ad1_design.png` |
| Figure A2 | `figures/fig_F2S.png` | `report/build_fig_F2S.py` | `build/figures/fig_F2S.png` |
| Figure A3 | `figures/fig_A_F3.png` | `report/build_fig_maps.py` | `build/figures/fig_A_F3.png` |
| Figure A4 | `figures/fig_A_F1.png` | `report/build_fig_A_F1.py` | `build/figures/fig_A_F1.png` |
| Figure A5 | `figures/fig_A_F2.png` | `report/build_fig_A_F2.py` | `build/figures/fig_A_F2.png` |
| Figure A6 | `figures/fig_A_F5.png` | `report/build_fig_A_F5.py` | `build/figures/fig_A_F5.png` |
| Figure A7 | `figures/fig_A_F7.png` | `report/build_fig_A_F7.py` | `build/figures/fig_A_F7.png` |
| Figure A8 | `figures/fig_A_F8.png` | `report/build_fig_A_F8.py` | `build/figures/fig_A_F8.png` |
| Figure A9 | `figures/fig_A_F4.png` | `report/build_fig_A_F4.py` | `build/figures/fig_A_F4.png` |
| Figure A10 | `figures/fig_ad3_city_rmse.png` | `report/build_fig_ad3_city_rmse.py` | `build/figures/fig_ad3_city_rmse.png` |
| Figure A11 | `figures/fig_ad3_strata.png` | `report/build_fig_ad3_strata.py` | `build/figures/fig_ad3_strata.png` |
| Figure A12 | `figures/fig_ad5_performance.png` | `report/build_fig_ad5_performance.py` | `build/figures/fig_ad5_performance.png` |
| Figure A13 | `figures/fig_ad6_persistence.png` | `report/build_fig_ad6_persistence.py` | `build/figures/fig_ad6_persistence.png` |
| Figure A14 | `figures/fig_ad7_density.png` | `report/build_fig_ad7_density.py` | `build/figures/fig_ad7_density.png` |
| Figure A15 | `figures/fig_ad7_classes.png` | `report/build_fig_ad7_classes.py` | `build/figures/fig_ad7_classes.png` |
| Figure A16 | `figures/fig_ad8_tail.png` | `report/build_fig_ad8_tail.py` | `build/figures/fig_ad8_tail.png` |
| Figure A17 | `figures/fig_ad4_cascade.png` | `report/build_fig_ad4_cascade.py` | `build/figures/fig_ad4_cascade.png` |
| Figure A18 | `figures/fig_ad4_square.png` | `report/build_fig_ad4_square.py` | `build/figures/fig_ad4_square.png` |
| Table 1 | &ndash; | (text of the report) | &ndash; |
| Table 2 | &ndash; | `report/build_tables.py` | `build/tex/tab_T2.tex` |
| Table 3 | &ndash; | `report/build_tables.py` | `build/tex/tab_T3.tex` |
| Table A1 | &ndash; | (text of the report) | &ndash; |
| Table A2 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T7.tex` |
| Table A3 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T1.tex` |
| Table A4 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T6.tex` |
| Table A5 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T3.tex` |
| Table A6 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T4.tex` |
| Table A7 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T8.tex` |
| Table A8 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T5.tex` |
| Table A9 | &ndash; | `report/build_tables.py` | `build/tex/tab_A_T2.tex` |

## Shared modules

The scripts above import these modules.

| Module | What it holds |
|---|---|
| `build_c2_common.py` | Helpers of the figure scripts: panel tags, number formats, the writers of the caption and alternative-text files, and the figure texts in reading order. |
| `build_c2_sealed.py` | Checks each named evaluation output against its folder's EVAL_DONE.json entry (sha256 and size) and stops on a missing file or a difference. |
| `build_check_text2.py` | Checks the text of a finished figure before it is saved: every visible text at least 7 pt, no overlapping text boxes, every box at least 0.08 in inside the canvas, and no line breaks; a failure stops the save. |
| `build_fig_common.py` | Shared helpers of the figure scripts: the style setup, the reading of evaluation outputs after their sha256 is checked, the text checks, color measures, and the layout of keys, ticks and blocks. |
| `build_mask_stream.py` | The land mask of the 12 km grid: the 53,594 land cells of the contiguous United States (299 rows by 459 columns, row 0 the south edge), read from results/grid/mask_conus_land.npy and tested against its sha256 before use. |
| `build_port_fig_A_F2.py` | The drawing part of figure fig_A_F2: the daily series panels, the daily differences of each U-Net from CMAQ, the month axis and the key. |
| `build_port_fig_A_F4.py` | The drawing part of figure fig_A_F4: the profile across wavelength bands of the two U-Nets and the per-cell network, then one panel per band with every network on its own row, on one log ratio axis. |
| `build_port_fig_A_F5.py` | The drawing part of figure fig_A_F5: the autocorrelation panels by season and lag, with one shared key and one line style per series. |
| `build_port_fig_A_F6.py` | The drawing part of figure fig_A_F6: the four models on the full scale, a dot at the five-year mean on a bar over the range of the five years, and the same values magnified. |
| `build_port_fig_A_F7.py` | The drawing part of figure fig_A_F7: the shares as markers for all land cells and each urban class, with a strip of gray tiles that gives each share in percent. |
| `build_port_fig_F2S.py` | The drawing part of figure fig_F2S: the two-way histograms of per-cell means on log axes with the 1:1 line, the cumulative curves of the differences, and their binned percentiles. |
| `build_port_fig_F3.py` | The drawing parts of figures fig_F3 and fig_A_F1: the city panels of daily series, the month labels and one key. |
| `build_port_fig_F5.py` | The drawing part of figure fig_F5: the skill scores of nine models, the seeds of the two U-Nets, RMSE by held-out year, and mean error against unbiased RMSE (the square root of RMSE squared minus mean error squared) for four models. |
| `build_port_fig_F6.py` | The drawing part of figure fig_F6: the five-year mean error by CMAQ decile with each U-Net's range over the years, CMAQ's decile bounds on the top axis, and the top decile of each held-out year. |
| `build_style.py` | The shared style of every figure: one sans-serif family at 7 and 8 pt, the colors, the panel letters, the text and layout checks, and the saving of a vector PDF, a 300 dpi PNG and the plotted values beside each figure. |
| `figure_style.py` | The font choice of the figure style: finds the installed font families and resolves the sans-serif family that the figures use. |
