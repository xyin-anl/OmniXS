- trunc of vasp and feff in range: 4960 + 40(or 45) eV:
  - fenchen did 35ev range..
- for feff to vasp alignment, use constant shift for now so that we do not
  introduce any uncertatinly
- [ ] refactor plot alignment based on new changes

- [x] make sure to shift all vasp (not feff) by : 5114.089730000899
- [x] try scipy constatn so that no can be hardcoded
- [x] make sure that 0.89 \* 2 to be for alignment comparision only
- [x] sperate alignment for vasp and feff to different class
- [x] make new file for vasp for average of mu named: xmu_avg.dat for all dataset.
- [x] for the test use broadening is 0.89 \*2 (ONLY FOR THE TEST)

## Urgent

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
