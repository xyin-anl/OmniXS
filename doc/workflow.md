# Workflow

## Featurization

````python

        # spectrum
        spectrum = ElemntSpectrum(
            element=Element.Cu,
            type=SpectrumType.FEFF,
            index=0,
            material= Material( id = ..., structure = ...),
            intensities=IntensityValues(...)
            energies=EnergyGrid(...),

        )

        # featurize material structure correspoinding to the spectrum
        features = M3GNetSiteFeaturizer().featurize(spectrum.material.structure, spectrum.index)
        ml_data = MLData(X=features, y=spectrum.intensities)

        # save the data
        FileHandler(cfg.serialization).serialize_json(
            ml_data,
            supplemental_info={
                **spectrum.dict(),
                "index_string": spectrum.index_string,
            },
        )
```
````

## Spectra Object

```python

        material_strucutre = MaterialStructure.from_file(poscar_path)
        material_id = MaterialID(id_string)
        material = Material(id=material_id, structure=material_strucutre)

        spectra_path = cfg.paths.processed_data.format(
            compound=element,
            simulation_type=spectra_type,
            id=id_string,
            site=site_string,
        )
        spectra_data = np.loadtxt(spectra_path)

        site_index = int(site_string)

        element_spectrum = ElementSpectrum(
            element=element,
            type=spectra_type,
            index=(
                site_index if spectra_type == SpectrumType.FEFF else 0
            ),  # coz sim was done this way
            material=material,
            intensities=IntensityValues(spectra_data[:, 1]),
            energies=EnergyGrid(spectra_data[:, 0]),
        )

        if spectra_type == SpectrumType.VASP:
            # bypass validation for saving because of the way the data was saved
            element_spectrum.__dict__.update(
                {"index": site_index}
            )  # TODO: remove in deployment

        save_path = (
            None
            if spectra_type == SpectrumType.FEFF
            else f"dataset/spectra/VASP/{element}/{id_string}_site_{site_string}_{element}_VASP.json"
        )  # hardcoded because of the way the data was saved TODO: fix this

        output.append(material)

        file_handler = FileHandler(config=cfg.serialization, replace_existing=False)
        file_handler.serialize_json(element_spectrum, custom_filepath=save_path)



```
