from __future__ import annotations

from dataclasses import dataclass

from tatg.models import GenerationRequest, PromptBundle, PromptInsertPosition, Target


class PromptPass:
    def apply(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        return prompt


@dataclass(slots=True)
class PromptPipeline:
    passes: list[PromptPass]

    def run(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        output = prompt
        for one_pass in self.passes:
            output = one_pass.apply(output, request, target)
        return output



def insert_text(prompt: PromptBundle, position: PromptInsertPosition, text: str) -> PromptBundle:
    if not text:
        return prompt

    sep = "\n\n"
    system = prompt.system
    user = prompt.user

    if position == PromptInsertPosition.BEFORE_SYSTEM:
        system = text + sep + system
    elif position == PromptInsertPosition.AFTER_SYSTEM:
        system = system + sep + text
    elif position == PromptInsertPosition.BEFORE_USER:
        user = text + sep + user
    else:
        user = user + sep + text

    return PromptBundle(system=system, user=user, metadata=dict(prompt.metadata))
