- do not forget to scale data before passing through fc models again
- finish things for LMBTR
- why is linreg and fc performaing similary for Cr:
- make sure the ckpt is the best and the last one
- make 1000 scale organized
- debug why fc is givin bad results
- refactor `scripts/model_scripts/plot_model_report.py`
- make visual rank better:
  - fourier in both rank and train loss
  - dtw
- Reviews of Modern Physics, Vol. 72, No. 3, July 2000
- histogram of residues in different basis and a kernet fit
- plot it based on the periodic table
- maybbe there is way to quantifiy coimplexiy of data
- later look at the receptive field as well
- qunatile plot
- what is wrong with MLP:
- maybe try the cu only prediciton and see how it maps with literature

  - if it doesnt work dropout and other regularization

- interpretation:
  - lin model with constraint so that product is positive and plot it
  - feature importance
  - see the percentile of the data ....
- [x] make the trained model load from the saved model
- [x] featurization using m3gnet:
  - [x] fix the bug in featurization
  - [x] featurize all feff data
- [ ] Establish baselines:
  - [ ] Previous results
  - [ ] MAD: Mean absolute deviation
  - [ ] MedAD: Median absolute deviation
  - [ ] Linear/Polynomial regression
- [ ] Uncertainty Quantification
- [ ] Find where the structure are coming form: DFT, experiment
  - [ ] Xiaohui is on it
- [x] Finish FEFF spectra processing:
  - [ ] Document data tranformation steps for FEFF
  - [x] Document Statistical report
- [ ] Finalize vasp data
  - [ ] Find broadening info for VASP
    - [ ] Deyu is working on it
- [x] decide how much data to take for each sim:
  - [ ] Then fix number of data for all
  - [ ] do full
  - [ ] Then common_id only
- [ ] for transfer learning do not use the alignement between feff and vasp.
      only that vasp is shifter xs comparision as is current before feff/vasp
      alignment
- [ ] use this code if it helps to make it fast:
      https://github.com/AI-multimodal/XAS-NNE

- [ ] benchmark after transformations
- [ ] how things changed?
  - [ ] better/worse?
  - [ ] architechtural differences
  - [ ] is vasp still overfitting?
  - [ ] are there training differences
- [ ] transfer learning

- [x] Throw away the data where sim has not converged:
  - [x] `grep "Convergence reached" */FEFF-XANES/*/feff.out | wc -l`
  - [x] For Mn it is `19575`
- [x] plot histogram of count of FEFF data
- [x] Find Cu experimental shift
  - [x] extract the shift from the Cu image:
- [x] process VASP data for mp-361
- [x] process raw_0 data:
  - [x] all FEFF data are already processed
- [x] fix VASP resampling error before truncation
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
- [x] make sure to shift all vasp (not feff) of Ti by : 5114.089730000899
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
