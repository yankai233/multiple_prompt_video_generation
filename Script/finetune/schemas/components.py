from typing import Any

from pydantic import BaseModel


class Components(BaseModel):
    # pipeline cls
    pipeline_cls: Any = None

    # Tokenizers
    tokenizer: Any = None
    tokenizer_2: Any = None
    tokenizer_3: Any = None

    # Text encoders
    text_encoder: Any = None
    text_encoder_2: Any = None
    text_encoder_3: Any = None

    # image encoder
    image_encoder: Any = None
    processor: Any = None

    # Autoencoder
    vae: Any = None

    # Denoiser
    transformer: Any = None
    unet: Any = None

    # transformer_block
    feature_extractor: Any = None

    # Scheduler
    scheduler: Any = None



