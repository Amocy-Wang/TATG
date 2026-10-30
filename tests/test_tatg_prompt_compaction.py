from tatg.engines.tatg_adapter import TATGEngineAdapter


def test_tatg_prompt_compaction_modes_are_gated(monkeypatch):
    high_priority = "[Active Uncovered Objectives]\n" + ("critical source line and oracle evidence. " * 120)
    appendix = "Source code under test:\n" + ("appendix context. " * 8000)
    prompt = high_priority + "\n\n---\n\n" + appendix

    monkeypatch.setenv("TATG_PROMPT_TARGET_CHARS", "16000")
    monkeypatch.setenv("TATG_PROMPT_COMPACTION_MODE", "off")
    assert TATGEngineAdapter._compact_user_prompt_to_budget(prompt) == prompt

    monkeypatch.setenv("TATG_PROMPT_COMPACTION_MODE", "protect_priority")
    compacted = TATGEngineAdapter._compact_user_prompt_to_budget(prompt)
    assert high_priority in compacted
    assert len(compacted) < len(prompt)
