- [x] decide how much data to take for each sim:
  - [ ] do full
  - [ ] Then common_id only
- [ ] upload after organizing
- [ ] for transfer learning do not use the alignement between feff and vasp.
      only that vasp is shifter xs comparision as is current before feff/vasp
      alignment
- [ ] use this code if it helps to make it fast:
      https://github.com/AI-multimodal/XAS-NNE
- [ ] extract the shift from the Cu image:

  - [ ] I think there is a mistake in the Cu vasp alignment. In the previous Ti
        alignment, after you did all the relative alignment within vasp, you
        shifted all the simulated spectra by a constant determined by aligning
        simulation to experiment for anatase TiO2, which we call the reference.
        This shift will be different for Cu K-edge, as it is a different
        element. Attached is an exp paper on Cu K-edge. Please digitize Cu2O
        spectrum in Fig. 1 and align [simulated](simulated) spectrum from
        mp-361 (Cu2O) to it, in order to get the correct value for the constant
        shift in Cu K-edge. Cu feff spectra are ok.

- [ ] benchmark after transformations
- [ ] how things changed?
  - [ ] better/worse?
  - [ ] architechtural differences
  - [ ] is vasp still overfitting?
  - [ ] are there training differences
- [ ] transfer learning

- [x] do multi-way paritioning implementation for train, test, val in material
      split. [original source_code](https://t.ly/cDSid).
- [x] multiway paritioning for train, test, val during material split
- [x] document data transformationprocess in markdown
- [x] confirm the dE starts fromm 4970 for eg
- [x] do constasnt trunctaion of 35 range for both Cu and Ti
- [x] decouple truncate_emperically
- [x] organize cfg
- [x] interpolate/extrapolate
- [x] include mvc data as well in raw data
- [x] run full compilations:

## Done for data transformation

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

- [x] Make feff alignment with VASP to be per spectra based on:

  - [x] correlation
  - [x] DTW:
  - [x] see where DTW and prev method disagree the most on the shift

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

## Unanswered Questions

- why was vasp overfitting?
- information leakage for m3gnet training?

- (paper)[https://www.nature.com/articles/s41467-022-30687-9]
