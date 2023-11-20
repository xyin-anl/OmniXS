## Talk these with Deyu

- [ ] double check enery filter from deyu:
  - This doesn't seem to be good looking at the plot
  - what was discussed in meeting: 4960 + 40(or 45) eV:
  - fenchen did 35ev range
  - what seems to be from plot is different
- [ ] for feff to vasp alignment, use constant shift for now so that we do not
      introduce any uncertatinly:
  - This makes a lot of difference in the alignment when compared with per
    spectra alignment

- [ ] try alignment using dynamic time wrapping

## Misc

- [ ] Do decile split
- [ ] m3gnet retrain prelims

## Done

- [x] Comparisions with MPNN:
  - [x] Do random split for FC and LR
- [x] refactor plot alignment based on new changes:
  - removed out. use bottom script in vasp_data_tranformations.py
- [x] make sure to shift all vasp (not feff) by : 5114.089730000899
- [x] try scipy constatn so that no can be hardcoded
- [x] make sure that 0.89 \* 2 to be for alignment comparision only
- [x] sperate alignment for vasp and feff to different class
- [x] make new file for vasp for average of mu named: xmu_avg.dat for all dataset.
- [x] for the test use broadening is 0.89 \*2 (ONLY FOR THE TEST)

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
