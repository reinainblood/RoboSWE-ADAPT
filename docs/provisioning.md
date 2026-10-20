# Prepare a fresh source checkout

RoboSWE's simulator images include the official RoboCasa and robosuite sources and kitchen assets. Git does not contain the downloaded kitchen assets. Run provisioning from your RoboSWE source checkout before building the images. Python 3.11, Git, and sufficient disk space for the six upstream asset groups, extraction, and container images are required.

```sh
roboswe prepare --source-root .
roboswe prepare --source-root . --run --out provisioning.json
roboswe build-images --source-root . -e docker --out images.json
```

The first command prints its plan without installing packages or downloading anything. The second clones the pinned official repositories into `upstream/`, creates `.roboswe-prepare-venv`, installs the simulation dependencies from `robot_session/requirements.txt`, installs the two source packages without training extras, and runs the upstream macro setup and asset downloader. The downloader retains its own interactive download confirmation. Run this command in a terminal and answer that prompt. RoboSWE does not accept any license terms on your behalf. The isolated environment is for downloading assets; container image building installs its own runtime.

For Modal images, replace the final command with:

```sh
roboswe build-images --source-root . -e modal --modal-profile YOUR_PROFILE --modal-environment FDR --out images.json
```

Use the immutable simulator and code image references from `images.json` in your adaptive run configuration. See [Adaptive CLI](adaptive-cli.md) for dataset, provider, and five-trial run configuration. Provisioning does not start a simulator or contact a model provider.

## Pinned source

- RoboCasa: `https://github.com/robocasa/robocasa.git`, commit `456174f62b89b8fca99eaaf33949c29fec9cfc2a`.
- robosuite: `https://github.com/ARISE-Initiative/robosuite.git`, commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.

These are the source revisions used by this release checkout. Asset URLs come from the pinned RoboCasa `box_links_assets.json`; the upstream command is `python -m robocasa.scripts.download_kitchen_assets --type all`. Its six asset groups cover textures, generated textures, fixtures, Objaverse objects, generated objects, and Lightwheel objects. The receipt records source revisions, completed commands, and whether every asset directory contains files. The upstream downloader does not supply cryptographic asset checksums, so directory presence is not an integrity certificate. The simulator reset and replay checks still run when tasks execute.

Existing upstream checkouts are inspected read-only. A matching pair with populated assets needs no installation. A mismatched, modified, partial, or missing-asset checkout is rejected instead of being reset or overwritten. After an interrupted download, retain its failed receipt and use a fresh RoboSWE source checkout for a new preparation attempt, or complete the official upstream installation yourself after reviewing its existing files. The automated command deliberately does not merge downloads into an existing checkout.

`provisioning.json` must be a new file. It records `planned`, `prepared`, or `failed`; a declined or unsuccessful download cannot produce `prepared`. The command does not delete partially downloaded files. Upstream installation instructions are at [RoboCasa installation](https://robocasa.ai/docs/build/html/introduction/installation.html).
