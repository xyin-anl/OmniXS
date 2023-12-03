Why is vasp overfitting?

**Universal XAS model**

- (paper)[https://www.nature.com/articles/s41467-022-30687-9]

**Truncation**

- [x] fix the tructaion problem that is putting things out of range

  - [x] issue was with offset not including theoretical offset

  **Final filter**

  - based on histogram of energy_start and energy_end
  - and meeting
  - take 40ev cutoff for both Cu and Ti:
  - energy_range = [4970, 4970 + 40] # Ti
  - energy_range = [4602, 4602 + 40] # Cu
  - implemented in `vasp_data.emperical_truncation()` that is not on by default

**Broadening**

- [x] refactor code so that vasp boradening happens based on the compound

**Alignment**

- [ ] Make feff alignment with VASP to be per spectra based on:

  - [ ] correlation
  - [ ] DTW:
  - [ ] see where DTW and prev method disagree the most on the shift

- [ ] make sure to store the shift values for each spectra:

**Transfer Learning**

- [ ] make it such that we can do feff -> vasp with and without the shift ( two
      models)
- [ ] Check the talk on domain adaptation which includes transfer learning and
      if it can be used for our task

**Misc**

- [ ] Do decile split
- [ ] m3gnet retrain prelims

## Notes

- Take broadening info from the table IIA in broadening table

- If vasp peak being consistently larger than feff a problem:

  - It seems like they are broadened differently
  - NOTE: that this affects the MAE the most in this region

- Fenchen did range of 35 ev:

  - So comparing MAE in different range will be different

- [x] for feff to vasp alignment, use constant shift for now so that we do not
      introduce any uncertatinly:
  - change of decition - make per spesctra shift but save the shifterd info as well so that it can be reproduced
  - This makes a lot of difference in the alignment when compared with per
    spectra alignment

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

## Code

- [x] icecream
