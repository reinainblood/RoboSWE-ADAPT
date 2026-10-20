# Website source

This directory contains the deployed RoboSWE-ADAPT homepage, styles, JavaScript,
provenance page, methods-preprint page, and small public JSON snapshots. These
files were checked against the public deployment when synchronized. Results are
timestamped snapshots, not a live connection to running jobs.

## Preview locally

From the repository root, inspect the optional asset inventory first. This makes
**no network requests** and writes no asset files:

```sh
python3 website/scripts/fetch-public-assets.py
```

For the complete local experience, explicitly retrieve the already-public videos
and selected episode viewers, then start a local static server:

```sh
python3 website/scripts/fetch-public-assets.py --download
python3 -m http.server 8891 --directory website
```

Open <http://127.0.0.1:8891/>. The homepage, preprint, videos, episode links,
trajectories and downloads then resolve locally. No provider key or simulator is
needed. Without the optional download, the text, catalogue, results snapshot and
preprint are available; video and episode assets have not been materialized.

The reviewed inventory is `public-assets.json`: 145 optional files totaling
440,114,098 bytes (about 420 MiB). Each file has its public HTTPS URL, exact byte
size and SHA-256. The helper permits only `https://roboswe.com`, validates paths,
rejects symlink destinations and redirects, enforces size bounds, and verifies
bytes before installing a file. Existing matching files are reused; an existing
mismatch is refused rather than overwritten. A changed deployment may require an
updated reviewed manifest. The helper never retrieves private job directories.

Large media and fetched episode directories are ignored by Git. They are not
silently redirected to production and are not included in this source commit.
Only curated episode exports already available on the public site are listed.
The static viewers cannot launch paid video judges.

## Verify the helper offline

```sh
python3 -m unittest discover -s website/tests -v
```

These tests use local byte fixtures and mocked HTTP responses, including invalid
paths, redirects, hash failures, symlink destinations and a no-network dry run.

## Preprint source

`paper/main.tex` at the repository root is the current canonical methods source;
`website/paper/source.tex` is its byte-identical published copy. With Pandoc
installed, regenerate the website from the canonical source:

```sh
python3 website/scripts/export-paper.py
```

The exporter does not add results. The HTML page also links the TeX and BibTeX and
supports browser print/save-as-PDF.

## Evidence and publication

The snapshot JSON and selected episode exports contain public, curated evidence.
Raw private jobs, credentials, `.vercel` account IDs, local preview screenshots,
and QA artifacts are excluded. The existing episode-export scripts require the
maintainer's original job evidence and are not needed to preview this site.

For deployment, serve `website/` as a static root after explicitly materializing
its reviewed assets. No installation or build step is required for serving.
