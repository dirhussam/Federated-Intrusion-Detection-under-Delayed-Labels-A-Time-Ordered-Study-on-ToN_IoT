# Releasing and obtaining a DOI (Zenodo)

Zenodo mints a DOI when a GitHub release is published on a repository that has
been enabled in Zenodo's GitHub settings.

## Before the first release

1. Replace every `TODO` in `CITATION.cff`, `README.md` and `data/README.md`
   (author name, ORCID, affiliation, data file, commands).
2. Validate the citation file: `pip install cffconvert && cffconvert --validate`.
3. Make the repository public (Zenodo archives public repositories).

## Enable the repository in Zenodo

1. Sign in at https://zenodo.org with the GitHub account that owns this
   repository.
2. Open the profile menu, choose **GitHub**, click **Sync now**.
3. Switch on this repository.

## Create the release

1. On GitHub: Releases → Draft a new release.
2. Tag: `v1.0.0`; title: `v1.0.0 — ICSIC 2026 camera-ready`.
3. Publish. Zenodo archives the release and assigns a DOI.

## After the release

1. Copy the **version DOI** of v1.0.0 from the Zenodo record.
2. Replace `XXXXXXX` in `README.md` and in `docs/CODE_AVAILABILITY.md`.
3. Add the DOI to `CITATION.cff` (`doi:` field) in a follow-up commit.

Notes:

- If both `.zenodo.json` and `CITATION.cff` exist, Zenodo uses only
  `.zenodo.json`. This repository uses `CITATION.cff` only.
- Zenodo records are intended to be permanent; check the content before
  publishing the release.
