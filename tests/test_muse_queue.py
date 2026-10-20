from pairing.muse_queue import disk_pause_reason,DISK_PAUSE


def test_disk_recovery_has_hysteresis():
    assert disk_pause_reason(None,7.9)==DISK_PAUSE
    assert disk_pause_reason(DISK_PAUSE,8.2)==DISK_PAUSE
    assert disk_pause_reason(DISK_PAUSE,8.5) is None
    assert disk_pause_reason(None,8.2) is None


def test_disk_recovery_never_clears_other_failure():
    assert disk_pause_reason('provider failure',20)=='provider failure'
    assert disk_pause_reason('provider failure',7)=='provider failure'
