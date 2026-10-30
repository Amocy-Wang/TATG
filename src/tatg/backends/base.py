from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from tatg.models import GenerationRequest, PromptBundle, Target


@dataclass(slots=True)
class CommandSpec:
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def build_commands(
        self,
        request: GenerationRequest,
        targets: list[Target],
        prompts: dict[str, PromptBundle],
    ) -> list[CommandSpec]:
        ...
