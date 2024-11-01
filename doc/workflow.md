# Workflow

## Featurization

`````python


    spectrum_type = SpectrumType.VASP
    for element in [Element.Cu]:
        spectra = DEFAULTFILEHANDLER().fetch_serialized_objects(
            ElementSpectrum,
            element=element,
            type=spectrum_type,
        )

        def save_ml_data(
            spectrum,
            file_handler=DEFAULTFILEHANDLER(),
            featurizer=M3GNetSiteFeaturizer(),
        ):
            index = 0 if spectrum.type == SpectrumType.VASP else spectrum.index
            features = featurizer.featurize(spectrum.material.structure, index)
            ml_data = MLData(X=features, y=np.array(spectrum.intensities))
            file_handler.serialize_json(
                ml_data,
                supplemental_info={
                    **spectrum.dict(),
                    "index_string": spectrum.index_string,
                },
            )

        for spectrum in tqdm(list(spectra), desc=f"Featurizing {element}"):
            save_ml_data(spectrum)
            ```


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

`````

## ML split

```python

    # omnixas.scripts.generate_ml_splits
    MLSplitGenerator().generate_ml_splits(
        DataTag(element=Element.Cu, type=SpectrumType.VASP),
        target_fractions=[0.8, 0.1, 0.1],
    )

```

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

```

```
