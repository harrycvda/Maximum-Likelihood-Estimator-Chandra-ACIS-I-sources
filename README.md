# Welcome to the WavLEX source code

WavLEX (Wavelet detection and Likelihood Evaluation of X-ray sources) is a Python tool for detecting X-ray sources in Chandra X-ray Observatory ACIS-I data, specialized for low exposures. It was developed by Van Der Ark et al. for the Chandra Legacy Survey.

For the full method and science results, see:

> Van Der Ark et al. 2026, "The Chandra Legacy Survey of the Central 6° × 4° around Sgr A*. I. WavLEX Source Detection Tool"
> [arXiv: DOI pending] · [ApJS: DOI pending]

This repository contains the self-contained WavLEX source code, `wavlex_v1.py`, an optional add-on for column density lookups, `nh_lookup.py`, and `launch_wavlex.py`, an optional helper for running several workers in parallel.

## Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Output](#output)
- [Optional: nH lookup](#optional-nh-lookup)
- [Limitations](#limitations)
- [Contact](#contact)

## Requirements

| Software | Version | Notes |
|---|---|---|
| CIAO | 4.17 or later (developed with CalDB 4.12.4) | Theoretically backwards compatible to CIAO 4.14 but untested |
| Python | 3.7 or later  | Included with CIAO |
| MARX | 5.5.3 or later | Included with CIAO |

Python and MARX come with the CIAO download, so a working CIAO installation is all most users need. WavLEX must be run in a shell where CIAO has been initialized. See the [CIAO download and installation guide](https://cxc.cfa.harvard.edu/ciao/download/) for help.

## Usage

Before running, open `wavlex_v1.py` and edit the `SETTINGS` block at the top:

| Setting | Variable | Description |
|---|---|---|
| ObsID(s) | `OBSIDS` | The Chandra ObsIDs to process, e.g. `[23884]` |
| Energy band | `E_MIN`, `E_MAX` | Minimum and maximum energy, in keV (default 0.5 to 8.0) |
| Output directory | `OUTDIR` | Folder where downloads, products and results are written (default `med_run`) |

Then run:

```bash
python wavlex_v1.py
```

WavLEX requires no pre-processing on the part of the user, raw ObsIDs will be used to download and process public Chandra X-ray observations directly into a source list.

The defaults (ObsID 23884, 0.5–8.0 keV) work as a first test. On these test settings a typical runtime 45 minutes on a single 4.05 GHz processing core and 300 MB required storage. The most computationally expensive process within WavLEX is fitting source models to each source candidate as such runtime is approximately proportional to the number of source candidates in a given ObsID. For users wishing to process several ObsIDs we recommend running parallel workers on a remote session.

To process many ObsIDs in parallel, set `N_WORKERS` in `wavlex_v1.py` and the same number in `launch_wavlex.py`, then run `python launch_wavlex.py`. Each worker takes an equal share of `OBSIDS`. With the default `N_WORKERS = 1`, run `wavlex_v1.py` directly. `N_WORKERS` must not exceed the number of cores on a users computer. 

## Output

A source list is written for each ObsID to the output directory:

```
{ObsID}_likelihood_results.csv
```

Main columns (see the paper for full definitions):

| Column | Description |
|---|---|
| `region_id` | Source number, in wavdetect order |
| `Wav_RA`, `Wav_DEC` | wavdetect position (deg) |
| `MLE_RA`, `MLE_DEC` | Position from the PSF fit (deg) |
| `Centroid_RA`, `Centroid_DEC` | Centroid of the counts in the fitted region (deg) |
| `Wav_2sigma_RA`, `Wav_2sigma_DEC` | wavdetect 2σ position errors (arcsec) |
| `Centroid_95_semimajor`, `Centroid_95_semiminor`, `Centroid_95_PA` | 95% error ellipse of the centroid (arcsec, arcsec, deg east of north) |
| `Centroid_N_events` | Events in the fitted region |
| `theta` | Off-axis angle (arcmin) |
| `region_counts`, `region_pixels/psf_size` | Counts and pixels in the fitted region |
| `bkg_count_density` | Background counts per pixel in the annulus around the source |
| `mean_region_exposure`, `normalised_exposure` | Mean exposure in the region, and relative to the image mean |
| `count_threshold` | Sources with fewer counts than this are classed `Too Faint` |
| `likelihood` | Maximum Likelihood Estimate from comparing source and background model, higher is more significant |
| `source_classification` | `True` (likelihood &geq; 10), `Marginal` (&geq; 2.5), `Unlikely`, `Too Faint`, or `nan` |
| `nh1`, `nh2` | Galactic HI and H₂ column densities (cm⁻²), NaN without `nh_lookup.py` |
| `photon_flux`, `photon_flux_2sig_minus`, `photon_flux_2sig_plus` | Photon flux and 2σ lower/upper errors (photons cm⁻² s⁻¹) |

Besides the csv, WavLEX keeps the download, the `chandra_repro` and `merge_obs` products, the final fit region of each source and the background map. All other intermediate files are deleted when an ObsID finishes.

## Optional: nH lookup

`nh_lookup.py` is an optional add-on that estimates the hydrogen column density (nH) toward each source. It was included because the Galactic Center fields WavLEX was developed on required it. Most users will not need it. Whether to include nH1 and nH2 estimates from public survey data is up to you, and you can adapt `nh_lookup.py` to your own fields.

It requires:

- The [HEASARC ftools package](https://heasarc.gsfc.nasa.gov/docs/software/ftools/)
- The CO intensity map `wco_map.fits` from [Dame et al. (2001)](https://doi.org/10.1086/318388), in the folder you run from (or set `WCO_MAP` in `wavlex_v1.py`)

Set `HEADAS_DIR` in `nh_lookup.py` to your HEASOFT install. If `nh_lookup.py` is not in the same folder as `wavlex_v1.py`, WavLEX skips the lookup and `nh1` and `nh2` are NaN.

## Limitations

- **Version 1 does not support merged or co-added observations.** WavLEX produces one source list per ObsID. If your observations overlap, you will need to crossmatch the source lists yourself.

## Contact

Questions and bug reports: please contact Harry Van Der Ark at hcv2109@columbia.edu.
