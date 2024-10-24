# Workflow

## Featurization

````python

        spectrum_1 = ElemntSpectrum(
            element=Element.Cu,
            type=SpectrumType.FEFF,
            index=0,
            material= Material( id = ..., structure = ...),
            intensities=IntensityValues(...)
            energies=EnergyGrid(...),

        )

        # featurize material structure correspoinding to the spectrum_1
        feature_1 = M3GNetSiteFeaturizer().featurize(spectrum_1.material.structure, spectrum_1.index)

        ...

        features = [feature_1, feature_2, ...]
        spectra = [spectrum_1.intensities, spectrum_2.intensities, ...]

        ml_data = MLData(X=features, y=spectra)

```
````

## Scaling

- `Warning`: Do not use scaler that makes the spectrum values negative if the models produce only positive values (e.g. `XASBlock` with `Softplus` activation !!)

## Training

```bash
    python -m refactor.model.training --config-path ../../config/training --config-name expertXAS element=Cu type=FEFF
```

Universal

```bash
python -m refactor.model.training --config-path ../../config/training --config-name universalXAS
```
