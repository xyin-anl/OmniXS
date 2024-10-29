# %%
from omnixas.utils import DEFAULTFILEHANDLER
from omnixas.data.constants import FEFFDataTags, VASPDataTags
from omnixas.data.ml_data import MLSplits


class VASPSplits:

    def __new__(cls, *args, **kwargs):
        return {
            tag.element: DEFAULTFILEHANDLER().deserialize_json(MLSplits, tag)
            for tag in VASPDataTags()
        }


class FEFFSplits:

    def __new__(cls, *args, **kwargs):
        return {
            tag.element: DEFAULTFILEHANDLER().deserialize_json(MLSplits, tag)
            for tag in FEFFDataTags()
        }


# %%
