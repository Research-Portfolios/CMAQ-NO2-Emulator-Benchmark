# Licenses for figures, results and text

Code in this repository (`pipeline/`, `report/` and the notebook's code cells) is released under the MIT license (`LICENSE`).

The figures in `figures/`, the derived values in `results/`, the analysis plan in `plan/`, the notebook's text and figures and the model weights attached to release v1.0.0 (the ten `best.pt` files listed in its notes) are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0), with the conditions below. Attribute to Ehsan Nikfarjam and Mohsen Nikfarjam, CMAQ-NO2-Emulator-Benchmark, and to the data sources below.

## Derived from EQUATES (public domain)

The evaluation outputs in `results/`, every figure and the model weights (trained on arrays made from EQUATES) derive from the U.S. Environmental Protection Agency's EQUATES v1.0 data, which are products of the U.S. Government in the public domain (EPA's data use policy, `results/EQUATES_Data_Use_Policy.pdf`). They are offered under CC BY 4.0 only as far as Ehsan Nikfarjam and Mohsen Nikfarjam hold rights in their arrangement. Cite the data as the policy asks: U.S. EPA (2021) for the datasets and Foley et al. (2023) for the emissions data (full entries in `README.md`).

## State lines on the maps (IPUMS terms)

The state lines, coast and borders drawn on these maps come from the 2010 TIGER/Line state boundaries of IPUMS NHGIS (Manson et al., 2024), used under the IPUMS terms with their citation. The boundary file itself is not included.

- `figures/fig_F2.png` (Figure 2)
- `figures/fig_F7.png` (Figure 9)
- `figures/fig_A_F3.png` (Figure A3)
- `figures/fig_ad6_persistence.png` (Figure A13)

## Natural Earth (public domain)

The land mask (`results/grid/mask_conus_land.npy`, with MCIP's land flag) and the city points were made with Natural Earth (admin-1 states v5.1.1 and populated places v5.1.2), which is in the public domain.

## Not covered

EPA's data use policy (`results/EQUATES_Data_Use_Policy.pdf`) remains EPA's document, outside this repository's licenses. The published values in `results/published/PUBLISHED_VALUES.txt` are quoted from Payami et al. (2025), with their pages, for comparison. The ORCID iD icon (`.github/orcid_id_icon.svg`) is ORCID's trademark, used as provided under ORCID's brand guidelines. No source dataset is redistributed.
