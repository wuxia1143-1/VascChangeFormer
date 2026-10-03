from __future__ import annotations

from dataclasses import dataclass

from .lac_v40_final import LACV40FinalConfig, LACiTransformerV40Final


@dataclass
class LACV42Config(LACV40FinalConfig):
    """Strict-no-tail V4.2 neural base used by all decision candidates."""

    progression_hurdle_enabled: bool = False
    hurdle_i_to_c_enabled: bool = False
    hurdle_risk_gate_enabled: bool = False
    hurdle_history_enabled: bool = False
    hurdle_treatment_enabled: bool = False


class LACiTransformerV42(LACiTransformerV40Final):
    """Protected TBR plus a private direct CAC path with no active tail stage."""

    architecture_version = "V4.2"

    def __init__(self, config: LACV42Config):
        config.progression_hurdle_enabled = False
        config.hurdle_i_to_c_enabled = False
        config.hurdle_risk_gate_enabled = False
        config.hurdle_history_enabled = False
        config.hurdle_treatment_enabled = False
        super().__init__(config)
        self.config = config
