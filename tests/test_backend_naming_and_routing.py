import pytest

from tatg.backends import get_backend
from tatg.backends.tatg_backend import TATGBackend


def test_get_backend_returns_tatg_backend():
    assert isinstance(get_backend("tatg"), TATGBackend)
    assert isinstance(get_backend("  TATG "), TATGBackend)


@pytest.mark.parametrize("name", ["mutgen", "method", "cat", "fusion", ""])
def test_get_backend_rejects_removed_backends(name):
    with pytest.raises(ValueError):
        get_backend(name)
