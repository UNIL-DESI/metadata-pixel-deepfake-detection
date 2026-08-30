from pathlib import Path
import numpy as np
import pandas as pd



def save_numpy(
        features,
        output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        output_path,
        features
    )



def save_csv(
        features,
        metadata,
        output_path
):

    output_path = Path(
        output_path
    )


    columns = [
        f"feature_{i}"
        for i in range(
            features.shape[1]
        )
    ]


    df_features = pd.DataFrame(
        features,
        columns=columns
    )


    df_metadata = pd.DataFrame(
        metadata
    )


    df = pd.concat(
        [
            df_metadata,
            df_features
        ],
        axis=1
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        output_path,
        index=False
    )