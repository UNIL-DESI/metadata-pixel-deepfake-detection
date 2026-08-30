from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


def get_cnn_model(model_name: str):

    model_name = model_name.lower()

    if model_name == "mobilenetv2":

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False
        )

    elif model_name == "efficientnetb0":

        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False
        )

    else:
        raise ValueError(
            f"Unsupported CNN model: {model_name}"
        )


    x = GlobalAveragePooling2D()(
        base_model.output
    )


    model = Model(
        inputs=base_model.input,
        outputs=x
    )


    model.trainable = False


    return model