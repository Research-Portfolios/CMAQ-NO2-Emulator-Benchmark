# Analysis plan

This is the public edition of the analysis plan of the report "Deep-learning emulators of CMAQ daily mean surface NO2 over the contiguous United States, 2015 to 2019" (Ehsan Nikfarjam and Mohsen Nikfarjam, 3 November 2026). The plan was written from 00:33 UTC on 28 October 2026 and fixed, with its sha256 recorded, at 00:34:50 UTC, before any evaluation output was opened, listed or copied. The safeguards came in this order: the study design was fixed before training; the two evaluations then wrote their outputs (made_utc in each eval_run.json), which were not opened before the plan was fixed; and only then were the outputs read, by this plan. This edition leaves out the references to files and records that are not part of this repository and spells out the abbreviations below or where they first appear; the analyses, their order, the comparisons and their margins are as fixed. The plan's own rule follows.

After a value is seen, nothing is added to, dropped from or reordered in this list; anything else the report says is labelled exploratory (or a record fact).

## Labels used in this plan

- The evaluations. CV: the first evaluation, on the test days of each held-out year (results/evaluation_test_years/ in this repository). LB: the second evaluation, which also evaluated the held-back days (results/evaluation_held_back/). OPN: the account of the single evaluation of the held-back days (not deposited). The lockbox: the held-back days, whole weeks set aside before any training. Fold y: the models trained without year y and tested on it; the forward fold is 2019, whose models saw only earlier years. s0, s1, ...: the training seed.
- The study design: the design of the study fixed before training; the evaluation code: the code that wrote the CV and LB outputs.
- The networks. M: the residual U-Net, the main emulator (its target is CMAQ minus the weekday climatology). P: the reference U-Net, built like the published one. A1 to A3: the three steps between P and M (A1, P with a squared-error loss; A2, residual blocks, added inputs and a logarithmic scale; A3, the residual target and the climatology input); A4: step 2 at the main training length. A3OFF: step 3 without the climatology input. A3BL: step 3 with one block of cells left out of training. M-C7: the residual U-Net with climatological emissions in place of each date's own. P18: P's form (one minus the index of agreement as loss, the direct target, no climatology channel) on the inputs of Salman et al. (2024). M ens5 and P ens5: the means of five seeds. PIX-R: the per-cell network.
- The further trainings. T1: A4 with 2019 held out. T2: M-C7 with 2019 held out. T3: further seeds with 2019 held out (M s3 and s4, P s2). T4: A3's form (residual target, climatology channel) trained on one year, with the climatology fitted in-sample on that year (three emission variants). T5: A3 at learning rates of 3e-4 and 3e-3.
- The baselines. GBT: gradient-boosted trees. GBT-ES: GBT at its round of lowest validation RMSE. GBT-RIDGE: the linear (ridge) model on GBT's features. MET-REG: the weather regression of each cell. CLIM-WD: the weekday climatology. CLIM-C: the cell mean. CLIM-EM: the emission-scaled climatology.
- C: the climatology channel (CLIM-WD read as an input). S_M: the main training length.
- The analyses. C-1 to C-4: the four comparisons (section 2 B). AN-02: the share of test cell-days with an emission input outside the range the cell saw in training. AN-05: M against M-C7 on 2016 and 2018. F-12: each model's weekday-weekend response against CMAQ's. D1 to D11: the diagnostics of section 2 K, each with its written direction.
- The statistics. RMSE: root mean square error; MB: mean bias, the model minus CMAQ (the report's mean error); MAE: mean absolute error; NMB and NME: normalized mean bias and error; R: Pearson correlation; R2: one minus the ratio of the squared error to the variance about CMAQ's mean; r_squared (r^2): the squared correlation; IOA: index of agreement; SS: skill score against CLIM-WD; ACC: anomaly correlation (of each cell's departures from CLIM-WD); MAPRMSE: the RMSE over cells of a yearly mean map; POD, FAR, CSI and ETS: the probability of detection, the false-alarm ratio, the critical success index and the equitable threat score; PBLH: boundary-layer height; Holm: Holm's step-down procedure.

## 1. The files read and their short names

**CV** the first evaluation's folder (see the labels above)

**LB** the second evaluation's folder

**OPN** the record of the lockbox's one opening

Files by name in each folder (as the evaluation code writes them): metrics.json, diagnostics.json, exceedance.json, deciles.json, band_ratios.json, residual_acf.json, d4_to_d7.json, family.json, exploratory.json, exploratory_strata.json, d11_curves.json, f12.json, an02.json, an05.json, gbt_es.json, five_fold.json, city_series.json, eval_run.json, EVAL_DONE.json (and each .csv twin; .npy arrays).

A key path is written file:key/key/...; [m,s,y] selects the row of model m, seed s (None for a baseline), fold y.

The reading source: every confirmatory, cv-day and diagnostic item is read from CV; the lockbox items from LB; item V-3 tests that LB's cv-day readings equal CV's. An item whose key is absent is reported as absent with the key tried; no other key replaces it.

Values as stored (float as JSON holds it; rounded only in the report's prose). Units as the source: RMSE, MB, MAE, MAPRMSE, F-12's contrast RMSE and the C-2 to C-4 differences in ppb (CMAQ surface NO2); SS, R, R2, r^2, IOA, NMB, NME, ACC, shares, slopes, ratios and counts unitless. No percent of a quantity that spans zero; like compared with like (the same statistic, level and days named).

## 2. The readings, in their fixed order

Tags: [CONF] confirmatory; [SENS] the family's pre-specified sensitivity; [SEED] the seed rule; [REPL] the forward fold and lockbox replication rule; [DESC] descriptive, no test; [DIAG] diagnostic with its written direction; [EXPL] exploratory; [CHK] a pre-specified check by a separate script of the reading; [REC] a record fact (no result).

### A. What ran (record facts, both evaluations)

**A-1** [REC] eval_run.json (CV and LB): run_id, made_utc, seconds, design, design_sha256, configs_sha256, models_sha256 (must equal the models table's, results/grid/MODELS_TIER_A.json), opening (LB: the opening record; CV: none), c7a_applies, resampling_seed (20261016), resamples (family 10000, exploratory 2000), test_resamples (must be null), not_run (listed whole), classes (urban and coast/edge cell counts per fold), guard_py, exploratory_margin_ppb (0.05), gbt_fit_checks, mc7_checks; the count of models by kind in "models" (86: 51 networks, 35 baselines) and of files_read by kind and part.

**A-2** [REC] the days and cells: for each fold, M s0's metrics.json row [M,0,y] stratum_kind "all": n and cells; n / cells = the CV test days (the study design: 344 or 345 in 2015-2018, 337 in 2019, inferred); LB family.json:lockbox/draws n_days per year (the study design: 21 a year in 2015-2018, 28 in 2019).

**A-3** [REC] EVAL_DONE.json (CV and LB): schema, run_id, made_utc, test_resamples, mc7_checks, the count of "files" (CV: as listed; LB: 263).

### B. The confirmatory family on the five-fold CV test days [CONF]

(m = 4; Holm as coded in the evaluation code; 10,000 moving-block resamples of 7-day blocks; margins C-1 0, C-2 to C-4 0.05 ppb; readings by the study design's rule, with +M and -M here the margin: wholly above +M "M closer to CMAQ", wholly below -M "the reference closer", inside [-M, +M] "the interval excludes differences larger than M in either direction", else "inconclusive at this precision"; a C-2 to C-4 half-width above 0.05 ppb "unable to resolve the margin")

**B-1** C-1, SS of M against CLIM-WD: CV family.json:five_fold/rows/C-1/{point, holm_rank, p_beyond, level, lo, hi, half_width, margin, passes_holm_step, reading, n_resamples, acc_median_five_fold, holds}. C-1 holds only if lo &gt; 0 and the five-fold median ACC &gt; 0 (the statement that it holds is written only if it holds).

**B-2** C-2, RMSE(GBT) - RMSE(M), ppb: .../rows/C-2/{the same keys but the ACC pair; resolution}.

**B-3** C-3, RMSE(P) - RMSE(M), ppb: .../rows/C-3/{as B-2}.

**B-4** C-4, MAPRMSE(CLIM-WD) - MAPRMSE(M), ppb: .../rows/C-4/{as B-2}.

**B-5** The yearly points of the four contrasts: CV family.json:five_fold/by_year/&lt;y&gt;/&lt;C-k&gt;, and the draws' records five_fold/draws (seed stream, block used, blocks per resample, n_days, n_resamples).

**B-6** status and count: five_fold/status, count_of_intervals_read.

### C. The seed rule [SEED]

(as coded: |point| &lt;= max - min of either model's seed values on the fold reads "not separated from training randomness")

**C-1s** 2016: CV family.json:per_fold rows fold 2016, C-1 to C-4: point, seed_rule/{status, ranges (M, and P for C-3: min, max, range, n_seeds), reading}.

**C-2s** 2019: the same rows for fold 2019.

**C-3s** the five-fold values with the 2016 ranges quoted: five_fold/rows/&lt;C-k&gt;/seed_2016_ranges_quoted.

**C-4s** 2015, 2017, 2018: per_fold rows of those folds, point and seed_2016_ranges_quoted.

**C-5s** C-3 on 2016: family.json:c3_seeds_2016/{n_seeds_M, n_seeds_P, mean_difference, sd_M, sd_P (sd, lo, hi, level, n_seeds)}.

**C-6s** [CHK] 2019 with the design's seed count: the study design names three seeds of M and two of P on 2019; T3 (M s3, s4 and P s2 on 2019, on the models table) was also trained, and the evaluation code's ranges use every seed on the table. A separate script recomputes the 2019 ranges from M s0-s2 and P s0-s1 only (metrics.json [M,k,2019] and [P,k,2019] stratum all: RMSE and SS; diagnostics.json rows: MAPRMSE) and applies the seed rule to the 2019 points of C-2s; reported beside the evaluation code's 2019 reading, labelled a check of the seed count.

### D. 2019 alone (the forward fold) [REPL; exploratory by the study design]

**D-1f** CV family.json:forward_2019/rows/&lt;C-k&gt;/{point, lo, hi, level, n_resamples, word, seed_rule, label} for C-1 to C-4 (95 percent, unadjusted, 2,000 resamples); count_of_intervals_read; draws.

### E. The 14-day block sensitivity [SENS]

**E-1** CV family.json:five_fold/rows/&lt;C-k&gt;/block_14/{lo, hi, level, reading, passes_holm_step} and five_fold/rows/&lt;C-k&gt;/block_length ("same reading" or "sensitive to the block length"); draws_14.

### F. The lockbox at its one opening [REPL]

**F-1** LB family.json:lockbox/rows/&lt;C-k&gt;/{point, lo, hi, level, n_resamples, word ("replicated" lo &gt; 0, "not contradicted" lo &lt;= 0 &lt;= hi, "contradicted" hi &lt; 0)}; lockbox/folds; draws; count_of_intervals_read; gbt_fit_checks; C-2's gbt_fit_not_equal_folds if present (a held word).

**F-2** The new-year statement: LB family.json:S6/&lt;C-k&gt; ("supported", "withdrawn" or "not read"); a "new year" sentence only where "supported".

**F-3** Shown beside the cv readings of B and D (same contrast; levels and days named: the five-fold Holm interval of B, the 2019 95 percent interval of D, the lockbox 95 percent interval of F-1).

**F-4** [REC] LB an02.json rows with part "LOCKBOX (apart)" (share, by urban class, n_cell_days, n_outside) and LB mc7_checks with part LOCKBOX: reported apart.

### G. The descriptive skill, all models [DESC]

(R2 about the mean of CMAQ's values)

**G-1** CV five_fold.json:rows (every model and seed with five folds): RMSE, MB, MAE, NMB, NME, R, R2, IOA, SS (five-fold means of yearly values). The intervals the study design names for this skill: reported as absent if no key of the evaluation outputs holds them.

**G-2** CV metrics.json rows stratum_kind "all", per fold, for every model and seed of the models table: n, cells, RMSE, MB, MAE, NMB, NME, R, R2, r_squared, IOA, SS (skill by fold; display 2).

**G-3** CV diagnostics.json rows (the per-model rows with RMSE): MAPRMSE, R_space, R_time median, ACC median, P10, P90, SS, share_SS_le_0, share_ACC_le_0, murphy_eq15 terms, per model, seed and fold.

**G-4** CV metrics.json rows by season, urban class and coast/edge class for the s0 members M, P, GBT, CLIM-WD (RMSE, MB, R, R2, n, cells).

**G-5** CV city_series.json rows (2016, M and P s0; per city: RMSE, MB, R, R2, IOA, cells, n).

### H. AN-02 (i), the emission inputs outside their trained range [DESC]

(networks only)

**H-1** CV an02.json rows part TEST_CV, every network row: fold, model, seed, name, source, channels, train_days, n_cell_days, n_outside, share, by_urban (share per class), outside_by_channel.

**H-2** M-C7's check (share 0 by construction): CV and LB mc7_checks (EVAL_DONE.json and eval_run.json): status and n_cell_days per fold and part; a FAIL is reported first, before any AN-02 reading.

**H-3** P18's AN-02 row: reported with its caveat (taken on the evaluation code's default "advanced" set, not P18's 19 inputs) and not used.

### I. AN-05, M against M-C7 on 2016 and 2018 [DESC, exploratory]

**I-1** CV an05.json rows: fold, row (M, M-C7, "M less M-C7"), status, n_cell_days, an02_share, f12_slope, words, reason; threshold. PASS of its pre-specified test when the rows print with their counts.

### J. F-12, the weekday-weekend response [EXPL]

(with the labels of J-2)

**J-1** CV f12.json rows: model, seed, fold, group (overall, urban, intermediate, rural), stat (R, slope, RMSE), point, lo, hi, finite_resamples, n_resamples, words; count_of_intervals_read; c7a_applies (whether the networks read each date's emissions).

**J-2** The labels: the evaluation code's words for every model. T4 (A3_F-6_TRUE, A3_F-6_OTHER, A3_F-6_OWN), T5 (A3_LR-3e-4, A3_LR-3e-3) and P18, whose rows carry no label in the evaluation code, are labelled here by the label rule's own test (does the network see CLIM-WD as offset or input?) on their forms: T4 and T5 "possibly carried by CLIM-WD's weekday factor (CLIM-WD their offset and C channel)", T4's C fitted in-sample on its one training year; P18 "emission-driven reading (no CLIM-WD in offset or input)"; because the networks read each date's emissions, every row also "one emission-driven variation among several". The baselines CLIM-C, CLIM-EM, CLIM-WD and GBT-ES keep the evaluation code's "not labelled".

**J-3** Report summary rows (every row stays in f12.json): group overall, R and slope with intervals, for M s0, P s0, GBT, CLIM-WD, MET-REG on five folds; A1 to A4 and M-C7 where trained; T4, T5, P18 with J-2's labels.

### K. The diagnostics [DIAG], each with its written direction, for every model on the table unless named

**K-1** D1 upper tail: CV deciles.json rows (decile 10: n, MB, MAE, RMSE, lower; and every decile for display 4: MB by decile for M s0 and P s0 on each fold); diagnostics.json above30 (level_ppb, n, R, IOA) for M and P per fold (expected: MB below 0 in decile 10 for every model; M's R above 30 ppb exceeds P's); exceedance.json at 13.30, 26.60, 63.83 ppb (cmaq_events, scored, POD, FAR, CSI, ETS) for M, P, GBT, CLIM-WD s0; diagnostics.json own_top_decile (MB, RMSE, overall_MB).

**K-2** D2 near-zero row: diagnostics.json D2_near_zero (count, by_class) (expected: near none for every model).

**K-3** D3 smoothing: diagnostics.json spread median; band_ratios.json rows (A1 against P on 2016 and 2019 by band; M and P) (expected: below 1 for all; lower for the MSE networks than for P at equal network, A1 against P).

**K-4** D4 transport beyond the field: d4_to_d7.json rows table D4 (season x urban: n, MB, MAE, RMSE, NME) for M and P s0 (expected: largest relative error in rural winter cells; M's rural winter error below P's).

**K-5** D5 coast and edge: rows table D5 (class, cells, RMSE, NME) for M, P, GBT, CLIM-WD s0 (expected: larger at coast and edge than interior for every model).

**K-6** D6 night boundary layer: rows table D6 (pblh_decile, RMSE, MB) for M and P s0 (expected: largest at the lowest PBLH decile).

**K-7** D7 missing sources: rows table D7 (season, urban, mean_residual_cmaq_minus_model) for M and P s0 (expected: positive in rural summer cells).

**K-8** D8 out of range: diagnostics.json D8 (cell_max bins and all, domain_max, inputs_outside_norm, inputs_outside: n, MB, RMSE, ratio_sum_e_over_o, n_days) for M and P s0 (expected: under-prediction there); P18's D8 with the caveat of H-3, not used.

**K-9** D9 year shift: CV five_fold.json:d9 rows (rel_change_rmse_2019, MB_by_year) per model and seed. The evaluation code's base is the mean RMSE of 2016, 2017 and 2018; the study design names 2015-2018 interior. [CHK] a separate script also computes (RMSE_2019 - mean RMSE 2015-2018) / that mean from metrics.json for every model with five folds, labelled a check of the evaluation code's base, not a reading of its own.

**K-10** D10 clip: diagnostics.json D10_clipped_masked per network.

**K-11** D11 under-training: diagnostics.json D11_summary (S_M trainings and 6,000-step trainings: still_improving of networks) and each network's D11_still_improving (expected: the 6,000-step trainings still improving at their end more often than the S_M trainings). d11_curves.json is hashed and its row count recorded; its validation values are not reported (a validation value is never a performance number).

**K-12** the residual autocorrelation: residual_acf.json first_below_0_2 by season (year, winter, spring, summer, fall) for M and P s0.

**K-13** The inference time: diagnostics.json G02_inference_seconds per network (a time, s).

**K-14** D3's and D2's figure arrays, F3, F4, F5: hashed with every file; read only for the displays named in section 3.

### L. The exploratory table [EXPL]

(95 percent, 2,000 resamples, 7-day blocks; each RMSE difference read at 0.05 ppb, words naming the closer model; the 2 x 2 at 0)

**L-1** CV exploratory.json rows in their stored order: the ladder on 2016 and 2019 (P -&gt; A1, A1 -&gt; A2, A2 -&gt; A3, A3 -&gt; M); "M-C7 stand-in" 2016, 2018 and 2019 (2019 = T2, the forward part of the stand-in's cost: exploratory); "A3OFF C input" 2016; "PIX-R context"; "CLIM-EM reference"; "A4 on 2019" (T1); "A3BL unseen place" 2016; "2 x 2 interaction" 2016 and 2019 (expected negative); "GBT-ES beside GBT" where printed; the ensembles on 2016 ("P ens5 - M ens5", "M s0 - M ens5"): each point, lo, hi, level, half_width, margin, reading, finite_resamples, n_resamples; count_of_intervals_read.

**L-2** CV gbt_es.json rows: fold, status, printed, worse_fraction, min_round, round, c2_model, reason (C-2 always reads GBT).

**L-3** CV exploratory_strata.json: count_of_intervals_read, not_run, and every row (fold, reference, stratum_kind, stratum, point, lo, hi, reading); the report names the count and the rows whose reading is not "inconclusive" only as a summary, labelled exploratory.

**L-4** The other further trainings (no contrast is pre-specified in the evaluation code): T4's six, T5's two and P18's metrics.json rows (stratum all: RMSE, MB, R, IOA, SS) beside A3's (2016) or P's (2016) rows, point values only, no interval, labelled exploratory.

### V. Checks of the reading itself [CHK]

(not results)

**V-1** every file's sha256 and bytes against its folder's EVAL_DONE.json "files"; the file set equal to the listed set plus EVAL_DONE.json.

**V-2** EVAL_DONE.json's sha256 of CV and LB against the completion markers the two evaluations wrote ("eval_done_sha256") and the lockbox opening record's cv_eval_done_sha256.

**V-3** LB's cv-day readings equal CV's: family.json five_fold, per_fold, c3_seeds_2016 and forward_2019; five_fold.json; metrics.json; exploratory.json (equal as stored; a difference is reported, and B to E stay read from CV).

**V-4** OPN: the second evaluation's run, one record (the opening claimed once).

## 3. The page's displays (at most four; data fixed now, form chosen later)

**DISP-1** the confirmatory readings with their intervals: B-1 to B-4 (point, Holm interval, margin), one panel per contrast unit (C-1 unitless; C-2 to C-4 ppb).

**DISP-2** skill by fold: G-2's RMSE (ppb) per fold for M s0, P s0, GBT, CLIM-WD (and M's 2016 and 2019 seed spread from G-2's seeds).

**DISP-3** the lockbox beside the cv: for each contrast, B's five-fold point and interval, D's 2019 point and interval and F-1's lockbox point and interval, the levels named.

**DISP-4** one diagnostic: K-1's MB by CMAQ decile for M s0 and P s0 (each fold, or the five folds' mean of the yearly values, said).

## 4. Known before any value (stated in the report as limits)

- The evaluation code's open items: the readings are the evaluation code's coded forms (Holm, the blocks across the lockbox gap, the seed rule, ACC, the lockbox, the margin of 0.05 ppb for the exploratory table, both rows of D8), which the study design does not state; the report says so.
- T4, T5 and P18 are labelled by J-2; P18's AN-02 and D8 are not used; the 35 baselines rest on the baseline stage's clean exit.
- The emulator takes EQUATES emission inputs; its response to changed emissions is untested and never claimed; A3 against A3OFF is exploratory.
- The lockbox confirms a direction, not the 0.05 ppb margin.
- What each interval covers.
