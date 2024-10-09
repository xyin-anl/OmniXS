from refactor.spectra_data import Spectrum
from refactor.spectra_enums import SpectraType


from pydantic import ValidationError


import unittest


class TestSpectrum(unittest.TestCase):
    def test_valid_spectrum(self):
        spectrum = Spectrum(
            type=SpectraType.VASP, energies=[1.0, 2.0, 3.0], intensities=[0.1, 0.2, 0.3]
        )
        self.assertEqual(spectrum.type, SpectraType.VASP)
        self.assertEqual(spectrum.energies, [1.0, 2.0, 3.0])
        self.assertEqual(spectrum.intensities, [0.1, 0.2, 0.3])

    def test_invalid_energies(self):
        with self.assertRaises(ValidationError):
            Spectrum(
                type=SpectraType.VASP,
                energies=[3.0, 2.0, 1.0],
                intensities=[0.1, 0.2, 0.3],
            )

    def test_negative_values(self):
        with self.assertRaises(ValidationError):
            Spectrum(
                type=SpectraType.VASP,
                energies=[-1.0, 2.0, 3.0],
                intensities=[0.1, 0.2, 0.3],
            )

    def test_unequal_lengths(self):
        with self.assertRaises(ValidationError):
            Spectrum(
                type=SpectraType.VASP, energies=[1.0, 2.0], intensities=[0.1, 0.2, 0.3]
            )