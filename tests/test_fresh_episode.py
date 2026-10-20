import os

from pairing.run_toaster_learning import FreshEpisode


def _hash_probe_worker(pipe, config, spec, directory, number, previous):
    pipe.send(
        {
            "ok": {},
            "fingerprint": str(hash("roboswe-reset-order")),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        }
    )
    request = pipe.recv()
    assert request["operation"] == "close"
    pipe.send({"ok": {"recording_dir": directory}})
    pipe.close()


def _probe(tmp_path, seed, index):
    episode = FreshEpisode(
        {"seed": seed},
        {"task_timeout_seconds": 1},
        tmp_path / str(index),
        index,
        [],
        worker=_hash_probe_worker,
    )
    try:
        return episode.initial
    finally:
        episode.close()


def test_fresh_episode_pins_spawned_python_hash_seed(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    first = _probe(tmp_path, 0, 1)
    second = _probe(tmp_path, 0, 2)
    other_seed = _probe(tmp_path, 1, 3)

    assert first["python_hash_seed"] == "0"
    assert second["python_hash_seed"] == "0"
    assert first["fingerprint"] == second["fingerprint"]
    assert other_seed["python_hash_seed"] == "1"
    assert other_seed["fingerprint"] != first["fingerprint"]
    assert "PYTHONHASHSEED" not in os.environ
