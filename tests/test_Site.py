from refactor.spectra_data import Site, Spectrum
from refactor.spectra_enums import Element, SpectraType


import unittest


class TestSite(unittest.TestCase):
    def test_valid_site(self):
        site = Site(index=0, element=Element.Fe)
        self.assertEqual(site.index, 0)
        self.assertEqual(site.element, Element.Fe)

    def test_assign_spectra(self):
        site = Site(index=0, element=Element.Fe)
        spectrum = Spectrum(
            type=SpectraType.VASP, energies=[1.0, 2.0], intensities=[0.1, 0.2]
        )
        site.assign_spectra(spectrum)
        self.assertEqual(site.spectrum, spectrum)

    def test_multiple_spectra(self):
        site = Site(index=0, element=Element.Fe)
        spectrum1 = Spectrum(
            type=SpectraType.VASP, energies=[1.0, 2.0], intensities=[0.1, 0.2]
        )
        spectrum2 = Spectrum(
            type=SpectraType.FEFF, energies=[1.0, 2.0], intensities=[0.1, 0.2]
        )
        site.assign_spectra(spectrum1)
        site.assign_spectra(spectrum2)
        with self.assertRaises(ValueError):
            _ = site.spectrum


if __name__ == "__main__":
    unittest.main()
