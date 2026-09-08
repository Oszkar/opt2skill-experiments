import os
from pathlib import Path

import pytest


def _assets_present() -> bool:
    root = Path(os.environ.get("O2S_THIRD_PARTY", Path.home() / "o2s_third_party"))
    return (root / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf").exists()


requires_assets = pytest.mark.skipif(not _assets_present(), reason="run scripts/fetch_assets.sh first")


@pytest.fixture(scope="session")
def cfg():
    from o2s.models import g1

    return g1.load_config()


@pytest.fixture(scope="session")
def mj_model(cfg):
    from o2s.models import g1

    return g1.load_mj_model(cfg)


@pytest.fixture(scope="session")
def pin_model(cfg):
    from o2s.models import g1

    return g1.load_pin_model(cfg)
