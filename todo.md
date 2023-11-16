- make new file for vasp for average of mu named: xmu_avg.dat for all dataset.
- scale feff by constant for transfer learning : maybe np.pi
- make sure to shift all vasp (not feff) by : 5114.089730000899

- for the test use broadening is 0.89 \*2 (ONLY FOR THE TEST)
- look at the spectre and the vasp should align with it
- for feff just match maximum

## Urgent

- [ ]do not forget to put all spectra in same range.fenchen did 35ev range..
- [ ] Rescale the VASP data on theoretical basis: - [ ] make the ids, mu and
      such matchup in wrangling code - Missing data: - RAW VASP data for Ti:
      3792, 92 (missing) - 3448 (non-zero), 399 (zero), 3847 (total) Lenght of
      Non-zero spectra: 3448 Lenght of Zero spectra: 399- - RAW VASP data for
      Cu: 3090, 227 (missing) - Location of full list : dataset/VASP-raw-data
- [ ] m3gnet retrain prelims

## Misc

- [ ] Do decile split

- [x] Comparisions with MPNN:

  - [x] Do random split for FC and LR

## Git

## Overreach

- [ ] the simplicity to advantage:

  - [ ] Any other metrics to judge linearity or complexity of problem in general
  - [ ] Interpretablity:
    - [ ] Which feature among 64 is siginficant
    - [ ] How does it map to the spectral property
    - [ ] Maybe other interpretable model than linear
  - [ ] What is amount of data needed ?
  - [ ] What other simplity advantage is there?

- [ ] Calculating the importance of M3gnet:
  - [ ] Do prediction using describ model without GNN and compare:
    - [ ] Important one is LMBTR
    - [ ] https://github.com/SINGROUP/dscribe
