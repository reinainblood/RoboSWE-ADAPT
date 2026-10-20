from collections import namedtuple

import pytest

from roboswe.storage import require_disk_headroom

Usage = namedtuple("Usage", "total used free")


def test_storage_checks_existing_parent_and_host_temp_volume(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    seen = []
    monkeypatch.setattr("roboswe.storage.tempfile.gettempdir", lambda: str(temp))

    def usage(path):
        assert path.exists()
        seen.append(path)
        return Usage(100 * 2**30, 80 * 2**30, 20 * 2**30)

    monkeypatch.setattr("roboswe.storage.shutil.disk_usage", usage)
    checks = require_disk_headroom(tmp_path / "new" / "jobs", 10)
    assert seen == [tmp_path, temp]
    assert len(checks) == 2


def test_low_host_temp_space_blocks_even_if_artifact_volume_has_room(
    tmp_path, monkeypatch
):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr("roboswe.storage.tempfile.gettempdir", lambda: str(temp))
    monkeypatch.setattr(
        "roboswe.storage.shutil.disk_usage",
        lambda p: Usage(100 * 2**30, 0, (0.5 if p == temp else 20) * 2**30),
    )
    with pytest.raises(RuntimeError, match="Insufficient host disk headroom"):
        require_disk_headroom(tmp_path)


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf")])
def test_invalid_storage_reserve_is_rejected(tmp_path, minimum):
    with pytest.raises(ValueError):
        require_disk_headroom(tmp_path, minimum)
