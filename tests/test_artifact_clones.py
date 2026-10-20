import sys
from pathlib import Path

import pytest

from roboswe.storage import clone_or_copy


def test_artifact_copy_preserves_bytes_and_independent_writes(tmp_path):
    source=tmp_path/'original.mp4';target=tmp_path/'viewer.mp4'
    source.write_bytes(b'original-evidence'*1000)
    result=clone_or_copy(source,target)
    assert source.read_bytes()==target.read_bytes()
    assert source.stat().st_ino!=target.stat().st_ino
    target.write_bytes(b'independent edit')
    assert source.read_bytes()==b'original-evidence'*1000
    assert result['method'] in ('copy_on_write','copy')
    if sys.platform=='darwin':assert result['method']=='copy_on_write'


def test_required_clone_failure_leaves_existing_destination(tmp_path,monkeypatch):
    import roboswe.storage as storage
    source=tmp_path/'source';target=tmp_path/'target'
    source.write_bytes(b'new');target.write_bytes(b'preserved')
    monkeypatch.setattr(storage.sys,'platform','linux')
    with pytest.raises(RuntimeError):clone_or_copy(source,target,require_clone=True)
    assert target.read_bytes()==b'preserved'
    assert not list(tmp_path.glob('.artifact-copy-*'))


def test_copy_fallback_and_same_path_guard(tmp_path,monkeypatch):
    import roboswe.storage as storage
    monkeypatch.setattr(storage.sys,'platform','linux')
    source=tmp_path/'source';target=tmp_path/'target';source.write_bytes(b'data')
    assert clone_or_copy(source,target)['method']=='copy'
    with pytest.raises(ValueError):clone_or_copy(source,source)


def test_retroactive_clone_preserves_existing_timestamp(tmp_path):
    import os
    source=tmp_path/'source';target=tmp_path/'target'
    source.write_bytes(b'evidence');target.write_bytes(b'evidence')
    os.utime(target,ns=(1_000_000_000,2_000_000_000))
    clone_or_copy(source,target,preserve_destination_metadata=True)
    assert target.stat().st_mtime_ns==2_000_000_000
    assert target.read_bytes()==b'evidence'
