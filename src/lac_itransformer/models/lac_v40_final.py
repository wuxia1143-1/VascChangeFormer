from __future__ import annotations

from dataclasses import dataclass

from .lac_v41 import LACV41Config, LACiTransformerV41


@dataclass
class LACV40FinalConfig(LACV41Config):
    """Final V4.0 candidate with an isolated direct CAC base and tail head."""

    use_private_cac_encoder: bool = True
    shared_transfer_enabled: bool = False
    historical_dose_enabled: bool = False
    cac_historical_inflammation_enabled: bool = False
    direction_magnitude_enabled: bool = False


class LACiTransformerV40Final(LACiTransformerV41):
    """Protected V3.7 TBR path plus a minimal, skippable CAC tail correction.

    The central CAC path is deliberately independent: it owns a private encoder,
    uses direct residual regression and cannot read historical inflammation via
    the shared transfer used by V3.7.  Strictly earlier inflammation and
    treatment may enter only through the detached progression-hurdle stage.
    Consequently, removing the hurdle returns the exact direct central path.
    """

    architecture_version = "V4.0-final"

    def __init__(self, config: LACV40FinalConfig):
        config.use_private_cac_encoder = True
        config.shared_transfer_enabled = False
        config.historical_dose_enabled = False
        config.cac_historical_inflammation_enabled = False
        config.direction_magnitude_enabled = False
        config.hard_phenotype_views = False
        config.cac_residual_adapter = False
        super().__init__(config)
        self.config = config
