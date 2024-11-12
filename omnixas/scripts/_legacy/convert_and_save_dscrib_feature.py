# %%


from omnixas.data import DataTag, MLData, MLSplits, MaterialSplitter
from omnixas.utils import DEFAULTFILEHANDLER


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def convert_and_save_dscrib_feature(tag):
    feature_name = tag.feature
    dscrbe_data = np.load(
        f"dataset/ML-data/{tag.element}_{feature_name}.npz"
    )  # TODO: remove hardcoding
    idSite = [(i, s) for i, s in zip(dscrbe_data["ids"], dscrbe_data["sites"])]
    train_ids, val_ids, test_ids = MaterialSplitter().split(
        idSite, target_fractions=[0.8, 0.1, 0.1]
    )

    def _to_tuple(data):
        return [tuple(d) for d in data]

    train_ids, val_ids, test_ids = (
        _to_tuple(train_ids),
        _to_tuple(val_ids),
        _to_tuple(test_ids),
    )

    features = dscrbe_data["features"]
    pca = PCA(n_components=0.99)
    scaler = StandardScaler()
    transformed_features = scaler.fit_transform(features)
    transformed_features = pca.fit_transform(features)

    data_dict = {
        (id, site): (feature, spectra)
        for id, site, feature, spectra in zip(
            dscrbe_data["ids"],
            dscrbe_data["sites"],
            transformed_features,
            dscrbe_data["spectras"],
        )
    }

    split = MLSplits(
        train=MLData(
            X=[data_dict[id][0] for id in train_ids],
            y=[data_dict[id][1] for id in train_ids],
        ),
        val=MLData(
            X=[data_dict[id][0] for id in val_ids],
            y=[data_dict[id][1] for id in val_ids],
        ),
        test=MLData(
            X=[data_dict[id][0] for id in test_ids],
            y=[data_dict[id][1] for id in test_ids],
        ),
    )

    DEFAULTFILEHANDLER().serialize_json(
        split,
        DataTag(element=tag.element, type=tag.type, feature=feature_name),
    )


# convert_and_save_dscrib_feature(
#     "SOAP",
#     DataTag(
#         element="Ni",
#         type="FEFF",
#         feature="SOAP",
#     ),
# )
