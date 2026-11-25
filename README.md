# WavLEX

**Wavelet detection and Likelihood Evaluation of X-ray sources**

WavLEX is a Python tool for detecting X-ray sources in Chandra X-ray Observatory observations, built on CIAO. It was developed by Van Der Ark et al. for the Chandra Legacy Survey of the Central 6° × 4° around Sgr A*.

For the full method and science results, see:

> Van Der Ark et al. (YEAR), "The Chandra Legacy Survey of the Central 6° × 4° around Sgr A*. I. WavLEX Source Detection Tool"
> [arXiv: DOI pending] · [ApJS: DOI pending]

This repository contains the self-contained WavLEX source code, `WavLEX_v1.py`, and an optional add-on for column density lookups, `nH_lookup.py`.

## Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Output](#output)
- [Optional: nH lookup](#optional-nh-lookup)
- [Limitations](#known-limitations)
- [Contact](#contact)

## Requirements

| Software | Version | Notes |
|---|---|---|
| CIAO | 4.17 (developed with CalDB 4.12.4) | The commands used are backward compatible with CIAO 4.14 |
| Python | 3.7 or later  | Included with CIAO |
| MARX | 5.5.3 or later | Included with CIAO |
| Python packages | |

Python and MARX come with the CIAO download, so a working CIAO installation is all most users need. WavLEX must be run in a shell where CIAO has been initialized. See the [CIAO download and installation guide](https://cxc.cfa.harvard.edu/ciao/download/) for help.

## Usage

Before running, open `WavLEX_v1.py` and set the following:

| Setting | Location | Description |
|---|---|---|
| ObsID(s) | line XXX | The Chandra ObsID(s) to process |
| Energy band | line YYY | Minimum and maximum energy, in keV |
| Output directory | line ZZZ | Path where results are written |

[TODO: replace line numbers with variable names, e.g. `OBSIDS`, `E_MIN`, `E_MAX`, `OUTDIR`.]

Then run:

```bash
python WavLEX_v1.py
```

[TODO: add a worked example with a sample ObsID and energy band, plus typical runtime and disk usage.]

WavLEX requires no pre-processing on the part of the user, raw ObsIDs will be used to download and process public Chandra X-ray observations directly into a source list.

## Output

A source list is written for each ObsID to the output directory:

```
{ObsID}_likelihood_results.csv
```

[TODO: add a short description of the columns, or point to the relevant table in the paper.]

## Optional: nH lookup

`nH_lookup.py` is an optional add-on that estimates the hydrogen column density (nH) toward each source. It was included because the Galactic Center fields WavLEX was developed on required it. Most users will not need it. Whether to include nH1 and nH2 estimates from public survey data is up to you, and you can adapt `nH_lookup.py` to your own fields.

It requires:

- The [HEASARC ftools package](https://heasarc.gsfc.nasa.gov/docs/software/ftools/)
- The CO intensity map `wco_map.fits` from [Dame et al. (2001)](https://doi.org/10.1086/318388), available in your local environment 

## Limitations

- **Version 1 does not support merged or co-added observations.** WavLEX produces one source list per ObsID. If your observations overlap, you will need to crossmatch the source lists yourself.

## Contact

Questions and bug reports: please contact Harry Van Der Ark at hcv2109@columbia.edu.
