# %%
from refactor.utilities.io import DEFAULTFILEHANDLER
from refactor.data.constants import FEFFDataTags, VASPDataTags
from refactor.data.ml_data import MLSplits


print("Loading splits...")
FEFFSplits = {
    tag.element: DEFAULTFILEHANDLER.deserialize_json(MLSplits, tag)
    for tag in FEFFDataTags
}
VASPSplits = {
    tag.element: DEFAULTFILEHANDLER.deserialize_json(MLSplits, tag)
    for tag in VASPDataTags
}

# %%
