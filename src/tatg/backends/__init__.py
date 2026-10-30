def get_backend(name: str):
    normalized = name.strip().lower()
    if normalized == "tatg":
        from tatg.backends.tatg_backend import TATGBackend
        return TATGBackend()
    raise ValueError(f"Unsupported backend: {name}")
