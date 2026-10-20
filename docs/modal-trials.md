# Modal trial infrastructure

The trusted simulator can run in a separate Modal sandbox in the **FDR**
environment. The model coordinator and its existing login remain local. Generated
Python still runs in the existing networkless Docker `CodeSandbox`, separate from
the simulator and its recording/verification state. This is simulator offload,
not a fully cloud-hosted coding harness.

## Reusable image and transport

`pairing/modal_episode.py build` creates an immutable image from the previously
validated RoboSWE base, installs the pinned simulation/video requirements, and
adds the current trusted runtime and worker modules. The build receipt is
`artifacts/modal-trials-image.json`. `RemoteEpisode` implements the `FreshEpisode`
coordinator interface through an SDK subprocess and a long-lived remote worker.
The SDK interpreter can be selected with `ROBOSWE_MODAL_PYTHON`; this workstation
uses its isolated uv Modal tool environment to avoid a legacy protobuf conflict
in the local scientific Python environment.

A fresh sandbox owns each episode. It currently reserves 4 CPUs, 16 GiB memory,
and an L4 GPU for EGL rendering. The FDR app is `roboswe-parallel-trials`. No model
credential or generated solver code is sent to this trusted simulator. Requests
are bounded coordinator operations; model output continues through CodeSandbox
validation and control-step limits. The remote sandbox has network egress blocked.

Every sandbox writes to a unique job/run directory on the persistent volume.
On close, recording files, physics snapshots and diagnostic evidence are copied
back to the local job directory before the owned sandbox is terminated. Existing
local artifacts are never overwritten. The jobs viewer continues to use its
usual local files. Failed startup sandboxes are also terminated.

## Runtime differences must remain visible

A measured 50-step recorded loop took approximately 67 seconds with CPU OSMesa
and 78 seconds with GPU rendering on the apple scene; transport is material.
The first migrated blender continuation therefore keeps its 50-step submission
limit and 3600-second active-work allowance, but uses a **120-second execution
wall limit**. This is an explicit infrastructure change, not a matched-runtime
comparison. Do not silently apply local execution timeouts to remote jobs.

`pairing/compare_backend_snapshots.py` compares initial snapshots without exposing
their privileged contents to the solver. It normalizes only known asset-root
paths and reports configuration, geometry, metadata, sensor and physical-state
differences separately. The blender migration matched geometry/task/seed, but
failed strict 1e-8 numeric equivalence across arm64 macOS and x86_64 Linux. The
apple position difference was about 1.54e-7 metres. The migration-specific
acceptance and both fingerprints are recorded in
`artifacts/blender-modal-transition.json`; the original strict comparison remains
in `artifacts/blender-modal-comparison.json`. Future trajectories are not assumed
to be bitwise identical.

## Persistent artifact backups

`pairing/modal_backup.py` inventories finalized job directories by default.
Add `--upload` to copy them into the **FDR** volume `roboswe-job-artifacts-v1`.
Use `--all-finalized` for the current collection, or pass specific job directories.
The CLI uses only standard Python plus the Modal SDK.

Each job uses a content-addressed path:

```
/jobs/<job-id>/snapshots/<manifest-sha256>/
```

All missing files are uploaded in a batch without overwrite permission. Every
remote file is read back and checked for byte count and SHA-256. A completion
manifest is published last, and a local verified receipt is saved beneath
`artifacts/modal-backups/`. Interrupted uploads can resume. Local files are kept;
no expiration or deletion policy is enabled.

The archive includes finalized top-level MP4s, timelines, reports, sources,
model responses, reasoning, context checkpoints, notes, libraries and snapshots.
Nested `recordings/` and `model-images/` copies are excluded to avoid redundant
storage. A background scan inventories the jobs present when it starts; newly
finished jobs should be submitted afterward by the separate readiness monitor.
