# %%
from typing import List
import numpy as np
from pydantic import Field, validate_call, BaseModel
from refactor.spectra_data import Spectrum


class OutlierRemover(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @validate_call
    def remove(
        self,
        spectra: List[Spectrum],
        std_factor: float = Field(gt=0),
    ) -> List[Spectrum]:
        spectra_numpy = np.array([s.intensities for s in spectra])
        mask = self.outliers_mask(spectra_numpy, std_factor)
        return [s for s, m in zip(spectra, mask) if m]

    @staticmethod
    def outliers_mask(
        array: np.ndarray,
        std_factor: float,
    ) -> np.ndarray:
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        upper_bound = mean + std_factor * std
        lower_bound = mean - std_factor * std
        mask = np.all((array <= upper_bound) & (array >= lower_bound), axis=1)
        return mask


# %%


if __name__ == "__main__":
    import unittest

    from refactor.spectra_data import IntensityValues, EnergyGrid

    class TestOutlierRemover(unittest.TestCase):  # TODO: move to test file
        def _random_spectrum(self):
            return Spectrum(
                intensities=IntensityValues(np.random.rand(5)),
                energies=EnergyGrid(np.arange(5)),
            )

        def test_outlier_removal(self):
            spectra = [self._random_spectrum() for _ in range(100)]
            intensites = np.array(spectra[0].intensities)
            intensites[0] += 10
            spectra[0].intensities = IntensityValues(intensites.tolist())
            spectra_cleaned = OutlierRemover().remove(spectra, std_factor=2)
            self.assertEqual(len(spectra_cleaned), len(spectra) - 1)

        def test_outlier_removal_no_outliers(self):
            spectra = [self._random_spectrum() for _ in range(100)]
            spectra_cleaned = OutlierRemover().remove(spectra, std_factor=2)
            self.assertEqual(len(spectra_cleaned), len(spectra))

    unittest.main(argv=[""], exit=False)

# %%
