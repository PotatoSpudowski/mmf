import torch

from mmf.common.registry import registry

from mmf.models.base_model import BaseModel

from mmf.utils.build import (
    build_encoder
)

from .albef.vit import AlbefVitEncoder
from .albef.xbert import AlbefBertEncoder

@registry.register_model("albef")
class Albef(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/concat_bert_tutorial/defaults.yaml"

    def build(self):
        self.vision_module = build_encoder(self.config.image_encoder)
        self.language_module = build_encoder(self.config.text_encoder)