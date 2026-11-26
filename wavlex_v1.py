"""
WavLEX v1: wavelet source detection and likelihood evaluation for Chandra ObsIDs.
Run from a CIAO shell with `python wavlex_v1.py`. Set the ObsIDs, energy band and
output folder in SETTINGS below. See README.md.

Per ObsID this keeps the download, repro/ (chandra_repro + merge_obs), the final fit
regions (repro/psf_fit_regions), the background map and the output csv. Everything
else is temporary and deleted when the ObsID finishes.
"""

# ==========================================================
# SETTINGS
# ==========================================================
OBSIDS = [23884]        # Chandra ObsIDs to process
E_MIN = 0.5             # energy band (keV)
E_MAX = 8.0
OUTDIR = "broadband_run"      # downloads, products and results all go here (avoid spaces in the path)

# optional
N_WORKERS = 1           # >1 only together with launch_wavlex.py, which needs the same number
KEEP_SCRATCH = False    # True keeps the temp files, for debugging

# nH lookup (needs nh_lookup.py)
WCO_MAP = "wco_map.fits"   # CO map from Dame et al. 2001
XCO = 0.53e20              # cm^-2 / (K km/s), depends on galactic position so calibrate for your field
NH_POSITION = "wav"        # position for the lookup: "wav" or "centroid"

# imports
import subprocess
import os
from pathlib import Path
import re
import csv
import math
import glob
import argparse
import atexit
import tempfile
import pandas as pd
import numpy as np
import shutil

# nh_lookup.py is optional, without it nh1 and nh2 are left as NaN
try:
    from nh_lookup import get_nh_estimates
    NH_LOOKUP_AVAILABLE = True
except ImportError as _nh_import_err:
    get_nh_estimates = None
    NH_LOOKUP_AVAILABLE = False
    print(
        f"[INFO] nh_lookup not available ({_nh_import_err}); "
        "nh1 / nh2 lookups will be skipped."
    )


def format_obsid(obsid):
    """5-digit, zero-padded ObsID string (123 -> '00123'), the form CIAO uses in file names."""
    try:
        n = int(str(obsid).strip())
    except ValueError:
        raise ValueError(f"ObsID must be a whole number, got {obsid!r}") from None
    if not 1 <= n <= 99999:
        raise ValueError(f"ObsID must be between 1 and 99999, got {obsid!r}")
    return f"{n:05d}"


def get_obsid_chunk(obsid_list, n_workers, worker_index):

    if worker_index < 1 or worker_index > n_workers:
        raise ValueError(
            f"iteration must be between 1 and {n_workers}, got {worker_index}"
        )

    # pad everything up front so a typo fails now, not hours into a run
    obsid_list = list(dict.fromkeys(format_obsid(o) for o in obsid_list))

    n = len(obsid_list)
    base, remainder = divmod(n, n_workers)

    chunks = []
    idx = 0
    for w in range(n_workers):
        size = base + (1 if w < remainder else 0)
        chunks.append(obsid_list[idx: idx + size])
        idx += size

    return chunks[worker_index - 1]


parser = argparse.ArgumentParser()
parser.add_argument(
    "iteration",
    nargs="?",
    default=1,
    type=int,
    help=f"1-indexed worker number (1-{N_WORKERS}). Selects which slice of "
         "OBSIDS this process handles. Optional (defaults to 1), so "
         "with N_WORKERS = 1 no argument is needed.",
)
args = parser.parse_args()

iteration = args.iteration

obsids = get_obsid_chunk(OBSIDS, N_WORKERS, iteration)

print(
    f"[Worker {iteration}] Assigned {len(obsids)} ObsIDs: "
    f"{', '.join(obsids)}"
)

tmpdir = tempfile.mkdtemp(prefix=f"iteration_{iteration}_")

# per-worker temp dir (pfiles, CIAO work area), deleted on exit
if not KEEP_SCRATCH:
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)

os.environ["TMPDIR"] = tmpdir
os.environ["ASCDS_WORK_PATH"] = tmpdir
os.environ["DM_CLOBBER"] = "yes"

# separate pfiles per worker so parallel runs don't clash

pfile_dir = os.path.join(tmpdir,"pfiles")
os.makedirs(pfile_dir,exist_ok=True)
nh_pfile_dir = os.path.join(tmpdir, "heasoft_pfiles")
old_pfiles = os.environ.get("PFILES", "")

SYSTEM_PFILES = old_pfiles.split(";")[-1] if old_pfiles else ""


def make_pfiles(user_dir):
    """PFILES string with `user_dir` as the writable user directory."""
    return f"{user_dir};{SYSTEM_PFILES}" if SYSTEM_PFILES else user_dir


os.environ["PFILES"] = make_pfiles(pfile_dir)


def get_theta(evt_file, asol_file, ra, dec):
    
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return np.nan

    call_pfile_dir = tempfile.mkdtemp(prefix="dmcoords_", dir=tmpdir)
    env = os.environ.copy()
    env["PFILES"] = make_pfiles(call_pfile_dir)

    try:
        result = subprocess.run(
            [
                "dmcoords",
                f"infile={evt_file}",
                f"asolfile={asol_file}",
                "opt=cel",
                "celfmt=deg",
                f"ra={ra}",
                f"dec={dec}",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(
                f"[WARN] dmcoords (theta) failed for ra={ra}, dec={dec}: "
                f"{result.stderr.strip()}"
            )
            return np.nan

        out = subprocess.run(
            ["pget", "dmcoords", "theta"],
            capture_output=True,
            text=True,
            env=env,
        )
        try:
            return float(out.stdout.strip())
        except ValueError:
            print(
                f"[WARN] could not parse dmcoords theta for ra={ra}, dec={dec}: "
                f"'{out.stdout.strip()}'"
            )
            return np.nan
    finally:
        shutil.rmtree(call_pfile_dir, ignore_errors=True)


base_dir = OUTDIR
os.makedirs(base_dir,exist_ok=True)

# CIAO imports

from astropy.io import fits
from astropy.table import Table
from sherpa.astro.ui import *
from scipy.special import gammaincc, gammainccinv, gammaln, exp1
from pycrates import read_file as crates_read_file  


def run_obsid(obsid):
    """Run one ObsID. Temp files go to a scratch folder that is deleted at the end."""
    scratch_dir = f"{base_dir}/scratch_{obsid}_{os.getpid()}"
    shutil.rmtree(scratch_dir, ignore_errors=True)    # start from a clean folder
    os.makedirs(scratch_dir)
    try:
        _run_obsid(obsid, scratch_dir)
    finally:
        if KEEP_SCRATCH:
            print(f"[INFO] KEEP_SCRATCH=True: temporary files kept in {scratch_dir}")
        else:
            shutil.rmtree(scratch_dir, ignore_errors=True)


def _run_obsid(obsid, scratch_dir):

    bin_size_scale = 0.5 # bin size = this * PSF semi-axis (geometric mean)
    smoothing_scale = 50 # sigma of the background smoothing (pixels)
    elo = E_MIN
    ehi = E_MAX
    eff = (ehi - elo)/(np.log(ehi) - np.log(elo))

    arcsec_per_pix = 0.492
    ci_nsigma = 2.0
    sigma2_arcsec_per_pix = ci_nsigma * arcsec_per_pix

    # frame for sherpa's psf.center: "physical" or "logical" (kernel pixels)
    psf_center_coords = "physical"
    

    # paths, no need to edit
    # kept: input_dir, repro_dir, psf_fit_dir, background_image, output_csv_path
    # everything under scratch_dir is temporary

    # phase 1
    # obsid is the padded 5-digit form used in file names. The download folder keeps
    # the unpadded name that download_chandra_obsid gives it (ObsID 123 -> 123/)
    download_id = str(int(obsid))
    input_dir = f"{base_dir}/{download_id}"
    repro_dir = f"{input_dir}/repro"

    evt2_file = f"{repro_dir}/acisf{obsid}_repro_evt2.fits"

    # merge_obs outputs
    img_file = f"{repro_dir}/{obsid}_{elo}-{ehi}_thresh.img"
    expmap_file = f"{repro_dir}/{obsid}_{elo}-{ehi}_thresh.expmap"
    psfmap_file = f"{repro_dir}/{obsid}_{elo}-{ehi}_thresh.psfmap"

    # wavdetect outputs
    srclist_fits = f"{scratch_dir}/{obsid}_sourcelist.fits"
    regfile = f"{scratch_dir}/{obsid}_wavdetect.reg"
    wav_scell = f"{scratch_dir}/{obsid}_scell.fits"    # wavdetect byproducts, deleted right after it runs
    wav_img = f"{scratch_dir}/{obsid}_img.fits"
    wav_defn = f"{scratch_dir}/{obsid}_defn.fits"

    # phase 3 (regions)

    input_region_file = regfile

    psf_image_template = (
        f"{scratch_dir}/{obsid}_PSF_{{region}}.psf"
    )

    psf_fit_dir = f"{repro_dir}/psf_fit_regions"        # kept
    psf_bkg_dir = f"{scratch_dir}/psf_bkg_regions"

    bin_size_path = os.path.join(scratch_dir, f"{obsid}_binsizes.csv")

    combined_psf_path = os.path.join(f"{psf_bkg_dir}/all_psf_regions.reg")

    # phase 3 (background)
    image_file = img_file
    all_region_file =f"{psf_bkg_dir}/all_psf_regions.reg"
    srclist = f"{psf_bkg_dir}/psf_smallbkg_regions.lis"
    bkglist = f"{psf_bkg_dir}/psf_bigbkg_regions.lis"
    background_image = f"{repro_dir}/{obsid}_background.img"

    # phase 3 (rebinning)
    input_image = img_file
    srcless_image = f"{scratch_dir}/{obsid}_{elo}-{ehi}_srcless.img"
    input_background = background_image
    input_emap = expmap_file
    input_psf_template = f"{scratch_dir}/{obsid}_PSF_{{region}}.psf"
    bin_sizes_csv = bin_size_path

    # phase 4

    image_tpl = f"{scratch_dir}/image_rebin_{{i}}.fits"
    srcless_tpl = f"{scratch_dir}/srcless_rebin_{{i}}.fits"
    bkgmap_tpl = f"{scratch_dir}/bkg_rebin_{{i}}.fits"
    emap_tpl = f"{scratch_dir}/emap_rebin_{{i}}.fits"

    region_bkg_tpl = f"{psf_bkg_dir}/psf_bigbkg_{{i:03d}}.reg"
    region_smallbkg_tpl = f"{psf_bkg_dir}/psf_smallbkg_{{i:03d}}.reg"
    region_src_tpl = f"{psf_fit_dir}/psf_fit_{{i:03d}}.reg"

    psf_tpl = f"{scratch_dir}/psf_rebin_{{i}}.fits"

    bin_csv_path = bin_size_path
    output_csv_path = f"{base_dir}/{obsid}_likelihood_results.csv"


    # ==========================================================
    # PHASE 1: download, reprocess, wavdetect
    # ==========================================================


    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # download

    if not Path(input_dir).exists():
        print(f"Downloading Chandra ObsID {obsid}...")
        subprocess.run(
            f"download_chandra_obsid {download_id}",
            cwd=base_dir,
            shell=True,
            check=True
        )
        print("Download complete.\n")
    else:
        print(f"ObsID {obsid} already exists. Skipping download.\n")

    # chandra_repro and merge_obs, skipped if the products already exist

    products_exist = (
        Path(img_file).exists()
        and Path(expmap_file).exists()
        and Path(psfmap_file).exists()
    )

    if products_exist:
        print("Merged products already exist. Skipping chandra_repro and merge_obs.\n")
    else:
        print("Missing merged products. Running chandra_repro and merge_obs...\n")

        commands = [
            [
                "chandra_repro",
                input_dir,
                repro_dir,
                "clobber=yes"
            ],
            [
                "merge_obs",
                evt2_file,
                f"{repro_dir}/{obsid}",
                f"bands={elo}:{ehi}:{eff}",
                "binsize=1",
                "psfecf=0.367",
                "psfmerge=expmap",
                "clobber=yes"
            ]
        ]

        for cmd in commands:
            print("\nRunning:")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

    asol_matches = sorted(glob.glob(f"{repro_dir}/pcadf*_asol1.fits"))
    asol_file = asol_matches[0] if asol_matches else None

    # wavdetect (output is temporary, so it runs every time)
    wavdetect_cmd = [
        "wavdetect",
        f"infile={img_file}",
        f"outfile={srclist_fits}",
        f"scellfile={wav_scell}",
        f"imagefile={wav_img}",
        f"defnbkgfile={wav_defn}",
        f"expfile={expmap_file}",
        f"psffile={psfmap_file}",
        f"regfile={regfile}",
        "scales=1.414 2 2.828 4 5.657 8 11.314 16 22.627 32 45.255 64 90.51 128",
        "sigthresh=1e-5",
        "clobber=yes"
    ]

    print("\nRunning wavdetect:")
    print(" ".join(wavdetect_cmd))
    subprocess.run(wavdetect_cmd, check=True)

    # scell, img and defn are only wavdetect byproducts
    for f in (wav_scell, wav_img, wav_defn):
        if Path(f).exists():
            os.remove(f)
            print(f"Deleted {f}")

    print("\nPreProcessing complete!")

    # ==========================================================
    # PHASE 2: simulate PSFs
    # ==========================================================

    # source list
    with fits.open(srclist_fits) as hdul:
        data = hdul[1].data

    # simulate_psf for each source
    for region_number, row in enumerate(data, start=1):

        ra = float(row["RA"])
        dec = float(row["Dec"])      # "DEC" if your column is uppercase

        outfile = f"{scratch_dir}/{obsid}_PSF_{region_number}"

        cmd = [
            "simulate_psf",
            f"infile={evt2_file}",
            f"outroot={outfile}",
            f"ra={ra}",
            f"dec={dec}",
            "spectrum=none",
            f"monoenergy={eff}",
            "flux=1e-2",
            "numrays=1000",
            "numiter=INDEF",
            f"asolfile={asol_file}",
        ]

        print(f"\nRunning source {region_number}")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"simulate_psf failed for source {region_number}")
            break

    print("Finished.")

    # ==========================================================
    # PHASE 3: PSF regions, background map, rebinning
    # ==========================================================

    # --- regions ---

    os.makedirs(psf_fit_dir, exist_ok=True)
    os.makedirs(psf_bkg_dir, exist_ok=True)

    for stale_path in [
        os.path.join(psf_bkg_dir, "psf_smallbkg_regions.lis"),
        os.path.join(psf_bkg_dir, "psf_bigbkg_regions.lis"),
    ]:
        if os.path.exists(stale_path):
            os.remove(stale_path)

    # matches ellipse(x,y,a,b,angle)
    pattern = re.compile(r'ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)')

    # bin size for each region
    bin_sizes = []

    # source mask ellipses (96% ECF) for the combined region file
    combined_psf_regions = []


    with open(input_region_file, "r") as infile:
        count = 1

        for line in infile:
            match = pattern.search(line)

            if match:
                # wavdetect ellipse center
                a = match.group(1).strip()
                b = match.group(2).strip()

                # PSF ellipses for this source
                with tempfile.TemporaryDirectory() as tmpdir:

                    PSF_file = psf_image_template.format(region=count)

                    # temp fits from dmellipse
                    psf_fit_fits = os.path.join(tmpdir, "fit.fits")
                    psf_bkg_fits = os.path.join(tmpdir, "bkg.fits")

                    # region files
                    psf_fit_reg = os.path.join(
                        psf_fit_dir,
                        f"psf_fit_{count:03d}.reg"
                    )

                    psf_bkg_reg = os.path.join(
                        psf_bkg_dir,
                        f"psf_bkg_{count:03d}.reg"
                    )
                    print(f"Using PSF file: {PSF_file}")

                    # fit region (80% ECF)
                    bash_cmd = (f"dmellipse {PSF_file} {psf_fit_fits} 0.8 clobber=yes")

                    result = subprocess.run(
                        ["bash", "-c", bash_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    print(
                        f"dmellipse stdout for count {count}:\n"
                        f"{result.stdout}"
                    )
                    print(
                        f"dmellipse stderr for count {count}:\n"
                        f"{result.stderr}"
                    )

                    full_cmd = (
                        f'dmmakereg "region({psf_fit_fits})" {psf_fit_reg} '
                        f"kernel=ascii clobber=yes"
                    )

                    result = subprocess.run(
                        ["bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    print(
                        f"dmmakereg stdout for count {count}:\n"
                        f"{result.stdout}"
                    )
                    print(
                        f"dmmakereg stderr for count {count}:\n"
                        f"{result.stderr}"
                    )

                    if os.path.exists(psf_fit_fits):
                        size = os.path.getsize(psf_fit_fits)
                        print(
                            f"psf_fit_fits {psf_fit_fits} "
                            f"size: {size} bytes"
                        )
                    else:
                        print(f"psf_fit_fits {psf_fit_fits} does not exist")

                    psf_pattern = re.compile(
                        r'ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)',
                        re.IGNORECASE
                    )

                    # read the ascii region from dmmakereg, not the fits from dmellipse
                    with open(psf_fit_reg, "r") as psf_infile:

                        found = False

                        for psf_line in psf_infile:

                            search = psf_pattern.search(psf_line)

                            if search:

                                found = True

                                f = float(search.group(3))
                                g = float(search.group(4))
                                h = search.group(5).strip()

                                # bin size from the PSF size
                                bin_size = max(
                                    1,
                                    round(bin_size_scale * math.sqrt(f * g))
                                )

                                print(
                                    f"Region {count}: "
                                    f"f={f}, g={g}, "
                                    f"bin_size={bin_size}"
                                )

                                bin_sizes.append((count, bin_size))

                                # fit region file
                                psf_text = (
                                    "# Region file format: DS9 version 4.1\n"
                                    "physical\n"
                                    f"ellipse({a},{b},{f},{g},{h})\n"
                                )

                                psf_path = os.path.join(
                                    psf_fit_dir,
                                    f"psf_fit_{count:03d}.reg"
                                )

                                with open(psf_path, "w") as psf_file:
                                    psf_file.write(psf_text)

                                print(f"Created {psf_path}")

                                ellipse_psf = (
                                    f"ellipse({a},{b},{f},{g},{h})"
                                )

                        if not found:
                            print(
                                f"No ellipse found in {psf_fit_reg}"
                            )

                    # background region (96% ECF)
                    bash_cmd = (f"dmellipse {PSF_file} {psf_bkg_fits} 0.96 clobber=yes")

                    result = subprocess.run(
                        ["bash", "-c", bash_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    print(
                        f"dmellipse stdout for count {count}:\n"
                        f"{result.stdout}"
                    )
                    print(
                        f"dmellipse stderr for count {count}:\n"
                        f"{result.stderr}"
                    )

                    # quotes protect the parentheses from bash
                    full_cmd = (
                        f'dmmakereg "region({psf_bkg_fits})" {psf_bkg_reg} '
                        f"kernel=ascii clobber=yes"
                    )

                    result = subprocess.run(
                        ["bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    print(
                        f"dmmakereg stdout for count {count}:\n"
                        f"{result.stdout}"
                    )
                    print(
                        f"dmmakereg stderr for count {count}:\n"
                        f"{result.stderr}"
                    )

                    if os.path.exists(psf_bkg_fits):
                        size = os.path.getsize(psf_bkg_fits)
                        print(
                            f"psf_bkg_fits {psf_bkg_fits} "
                            f"size: {size} bytes"
                        )
                    else:
                        print(f"psf_bkg_fits {psf_bkg_fits} does not exist")

                    psf_pattern = re.compile(
                        r'ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)',
                        re.IGNORECASE
                    )

                    # read the ascii region from dmmakereg, not the fits from dmellipse
                    with open(psf_bkg_reg, "r") as psf_infile:

                        found = False

                        for psf_line in psf_infile:

                            search = psf_pattern.search(psf_line)

                            if search:

                                found = True

                                i = float(search.group(3))
                                j = float(search.group(4))
                                k = search.group(5).strip()

                                # source mask region (96% ECF)
                                psf_text = (
                                    "# Region file format: DS9 version 4.1\n"
                                    "physical\n"
                                    f"ellipse({a},{b},{i},{j},{k})\n"
                                )

                                psf_path = os.path.join(
                                    psf_bkg_dir,
                                    f"psf_smallbkg_{count:03d}.reg"
                                )

                                with open(psf_path, "w") as psf_file:
                                    psf_file.write(psf_text)

                                print(f"Created {psf_path}")

                                ellipse_psf = (
                                    f"ellipse({a},{b},{i},{j},{k})"
                                )

                                combined_psf_regions.append(ellipse_psf)

                                psf_lis_path = os.path.join(
                                    psf_bkg_dir,
                                    "psf_smallbkg_regions.lis"
                                )

                                with open(
                                    psf_lis_path,
                                    "a"
                                ) as psf_lis_file:
                                    psf_lis_file.write(
                                        f"{ellipse_psf}\n"
                                    )

                                # background annulus
                                i_small = 1.1 * i
                                j_small = 1.1 * j

                                if i < 10 or j < 10:
                                    i_big, j_big = 50, 50
                                else:
                                    i_big, j_big = i * 5, j * 5

                                psf_bkg_text = (
                                    "# Region file format: DS9 version 4.1\n"
                                    "physical\n"
                                    f"ellipse({a},{b},{i_big},{j_big},{k})"
                                    f"-ellipse({a},{b},{i_small},{j_small},{k})"
                                )

                                psf_bkg_path = os.path.join(
                                    psf_bkg_dir,
                                    f"psf_bigbkg_{count:03d}.reg"
                                )

                                with open(
                                    psf_bkg_path,
                                    "w"
                                ) as psf_bkg_file:
                                    psf_bkg_file.write(psf_bkg_text)

                                print(f"Created {psf_bkg_path}")

                                ellipse_psf_bkg = (
                                    f"ellipse({a},{b},{i_big},{j_big},{k})"
                                    f"-ellipse({a},{b},{i_small},{j_small},{k})"
                                )

                                psf_bkg_lis_path = os.path.join(
                                    psf_bkg_dir,
                                    "psf_bigbkg_regions.lis"
                                )

                                with open(
                                    psf_bkg_lis_path,
                                    "a"
                                ) as psf_bkg_lis_file:
                                    psf_bkg_lis_file.write(
                                        f"{ellipse_psf_bkg}\n"
                                    )

                        if not found:
                            print(
                                f"No ellipse found in "
                                f"psf_bkg_reg {psf_bkg_reg}"
                            )

                count += 1

    # bin sizes csv

    with open(
        bin_size_path,
        "w",
        newline=""
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(
            ["Region Number", "Bin Size"]
        )

        writer.writerows(bin_sizes)

    print(
        f"Bin sizes written to "
        f"{bin_size_path}"
    )

    # combined region file
    with open(combined_psf_path, "w") as outfile:

        outfile.write(
            "# Region file format: DS9 version 4.1\n"
        )
        outfile.write("physical\n")

        for reg in combined_psf_regions:
            outfile.write(reg + "\n")

    print(
        f"Combined PSF region file written to "
        f"{combined_psf_path}"
    )

    # --- background map ---
    def run_cmd(cmd):
        """Run a shell command and stop if it fails."""
        print(f"\nRunning:\n{cmd}\n")
        subprocess.run(cmd, shell=True, check=True)


    def main():

        # temp files
        temp_copy = tempfile.NamedTemporaryFile(
            suffix=".fits", delete=False
        ).name

        temp_dmfilth = tempfile.NamedTemporaryFile(
            suffix=".fits", delete=False
        ).name

        try:

            # mask out the sources
            cmd = (
                f'dmcopy "{image_file}[exclude sky=region({all_region_file})]" '
                f'outfile={srcless_image} clobber=yes'
            )
            run_cmd(cmd)

            # fill the holes with poisson noise
            cmd = (
                f"dmfilth infile={srcless_image} "
                f"outfile={temp_dmfilth} "
                f"method=POISSON "
                f"srclist=@{srclist} "
                f"bkglist=@{bkglist} "
                f"clobber=yes"
            )
            run_cmd(cmd)

            # smooth
            kernel = (
                f'"lib:gaus(2,5,1,{smoothing_scale},{smoothing_scale})"'
            )

            cmd = (
                f"aconvolve infile={temp_dmfilth} "
                f"outfile={background_image} "
                f"kernelspec={kernel} "
                f"method=fft "
                f"clobber=yes"
            )
            run_cmd(cmd)

            print("\nBackground image created successfully!")
            
        finally:
            # cleanup
            for f in [temp_copy, temp_dmfilth]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Deleted temporary file: {f}")

    if __name__ == "__main__":
        main()
        
    # global background rate
    cmd = ["dmstat", f"{srcless_image}", "centroid=no"]

    result1 = subprocess.run(cmd, capture_output=True, text=True)

    sum_match1 = re.search(r"sum:\s+([0-9eE+.-]+)", result1.stdout)
    good_match1 = re.search(r"good:\s+([0-9eE+.-]+)", result1.stdout)

    sum_val1 = float(sum_match1.group(1))
    good_val1 = float(good_match1.group(1))
    global_bkg_count_rate = (sum_val1 / good_val1)

    print(f"Global Background Rate = {global_bkg_count_rate} counts/pixel")
        
    # --- rebinning ---

    # bin sizes
    df = pd.read_csv(bin_sizes_csv)
    region_numbers = df['Region Number'].tolist()
    bin_sizes = df['Bin Size'].tolist()

    def get_image_dimensions(fits_file):
        cmd = ['dmlist', fits_file, 'blocks']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if '(' in line and 'x' in line:
                try:
                    dims = line.split('(')[1].split(')')[0]
                    xmax, ymax = dims.split('x')
                    return int(xmax), int(ymax)
                except Exception:
                    continue
        raise ValueError(f"Could not parse image dimensions for {fits_file}")

    # main image size
    main_xmax, main_ymax = get_image_dimensions(input_image)

    # rebinned files per bin size, so each size is only done once
    processed_bins = {}

    for region, bin_size in zip(region_numbers, bin_sizes):
        bin_size_int = int(round(bin_size))

        region_outputs = {
            'image': f'{scratch_dir}/image_rebin_{region}.fits',
            'bkg': f'{scratch_dir}/bkg_rebin_{region}.fits',
            'emap': f'{scratch_dir}/emap_rebin_{region}.fits',
            'srcless': f'{scratch_dir}/srcless_rebin_{region}.fits'
        }

        psf_outfile = f'{scratch_dir}/psf_rebin_{region}.fits'

        # skip if already done
        all_outputs = list(region_outputs.values()) + [psf_outfile]

        if all(os.path.exists(f) for f in all_outputs):
            print(f"[SKIP] Region {region}: all rebinned files already exist.")
            continue
        # image, background, emap, srcless
        if bin_size_int == 1:
            # bin size 1: just copy
            for infile, outfile in zip([input_image, input_background, input_emap, srcless_image],
                                    region_outputs.values()):
                print(f"[COPY] Bin size = 1 for region {region}, copying {infile} -> {outfile}")
                shutil.copy(infile, outfile)
        else:
            if bin_size_int in processed_bins:
                # reuse the cached rebin
                print(f"[CACHE] Bin size {bin_size_int} already processed. Copying for region {region}.")
                for key, cached_file in processed_bins[bin_size_int].items():
                    shutil.copy(cached_file, region_outputs[key])
            else:
                # new bin size
                print(f"[REGRID] Bin size {bin_size_int} new. Rebinning main files for region {region}.")
                bin_expr = f"1:{main_xmax}:{bin_size},1:{main_ymax}:{bin_size}"

                # cache files
                cache_outputs = {
                    'image': f'{scratch_dir}/image_bin_{bin_size_int}.fits',
                    'bkg': f'{scratch_dir}/bkg_bin_{bin_size_int}.fits',
                    'emap': f'{scratch_dir}/emap_bin_{bin_size_int}.fits',
                    'srcless': f'{scratch_dir}/srcless_bin_{bin_size_int}.fits'
                }

                main_files = [
                    (input_image, cache_outputs['image']),
                    (input_background, cache_outputs['bkg']),
                    (input_emap, cache_outputs['emap']),
                    (srcless_image, cache_outputs['srcless'])
                ]

                for infile, outfile in main_files:
                    cmd = [
                        'dmregrid', infile, outfile,
                        bin_expr,
                        'rotangle=0', 'rotxcenter=0', 'rotycenter=0',
                        'xoffset=0', 'yoffset=0', 'npts=0',
                        'clobber=yes'
                    ]
                    print(f"[REGRID] Running: {' '.join(cmd)}")
                    subprocess.run(cmd)

                processed_bins[bin_size_int] = cache_outputs

                for key, cached_file in cache_outputs.items():
                    shutil.copy(cached_file, region_outputs[key])

        # psf (one per source)
        psf_infile = input_psf_template.format(region=region)
        psf_outfile = f'{scratch_dir}/psf_rebin_{region}.fits'

        if os.path.exists(psf_infile):
            psf_xmax, psf_ymax = get_image_dimensions(psf_infile)
            psf_bin_expr = f"1:{psf_xmax}:{bin_size},1:{psf_ymax}:{bin_size}"

            if bin_size_int == 1:
                print(f"[COPY] Bin size = 1 for region {region}, copying {psf_infile} -> {psf_outfile}")
                shutil.copy(psf_infile, psf_outfile)
            else:
                cmd = [
                    'dmregrid', psf_infile, psf_outfile,
                    psf_bin_expr,
                    'rotangle=0', 'rotxcenter=0', 'rotycenter=0',
                    'xoffset=0', 'yoffset=0', 'npts=0',
                    'clobber=yes'
                ]
                print(f"[REGRID-PSF] Running: {' '.join(cmd)}")
                subprocess.run(cmd)
        else:
            print(f"[WARNING] PSF file missing for region {region}: {psf_infile}")

    # ==========================================================
    # PHASE 4: MLE fitting
    # ==========================================================

    # fresh sherpa session
    clean()

    # results
    with fits.open(srclist_fits) as hdul:
        number_sources = len(hdul[1].data)
        
    bkg_amps = []
    bkg_count_rates = []
    region_counts = []
    region_pixels = []             
    mean_region_exposure = []   
    normalised_exposure = []       
    count_thresholds = []      
    bkg_fit = []
    emap_region_vals = []
    src_fit = []
    dof = []
    corrected_ra = np.full(number_sources, np.nan)
    corrected_dec = np.full(number_sources, np.nan)

    region_x_spans = np.full(number_sources, np.nan)
    region_y_spans = np.full(number_sources, np.nan)

    # bin size per region
    bin_map = {}
    with open(bin_csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            bin_map[int(row[0])] = int(row[1])

    # helpers
    def read_region_center(region_file):
        with open(region_file) as f:
            for line in f:
                if line.strip().startswith("ellipse"):
                    vals = line.strip().split("(")[1].split(")")[0]
                    parts = vals.split(",")
                    return float(parts[0]), float(parts[1])
        return None, None

    def log_gamma_ratio(s, x):
        if s == 0:
            return np.log(exp1(x))
        return np.log(gammaincc(s, x))
    
    def update_region_center(region_file, new_x, new_y):
        pattern = re.compile(
            r'ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)',
            re.IGNORECASE
        )

        with open(region_file, "r") as f:
            content = f.read()

        match = pattern.search(content)
        if not match:
            print(f"[WARN] No ellipse found in {region_file}; center not updated.")
            return False

        f_val, g_val, h_val = match.group(3), match.group(4), match.group(5)

        new_text = (
            "# Region file format: DS9 version 4.1\n"
            "physical\n"
            f"ellipse({new_x:.6f},{new_y:.6f},{f_val},{g_val},{h_val})\n"
        )

        with open(region_file, "w") as f:
            f.write(new_text)

        print(f"[UPDATE] {region_file}: center -> ({new_x:.4f}, {new_y:.4f})")
        return True

    def pix_to_radec(x, y):
        cmd = [
            "dmcoords",
            f"infile={evt2_file}",
            f"asolfile={asol_file}",
            "opt=sky",
            "celfmt=deg",
            f"x={x}",
            f"y={y}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] dmcoords failed for x={x}, y={y}: {result.stderr}")
            return np.nan, np.nan

        ra_result = subprocess.run(
            ["pget", "dmcoords", "ra"], capture_output=True, text=True
        )
        dec_result = subprocess.run(
            ["pget", "dmcoords", "dec"], capture_output=True, text=True
        )

        try:
            ra = float(ra_result.stdout.strip())
            dec = float(dec_result.stdout.strip())
        except ValueError:
            print(
                f"[WARN] Could not parse dmcoords output for x={x}, y={y}: "
                f"ra='{ra_result.stdout.strip()}' dec='{dec_result.stdout.strip()}'"
            )
            ra, dec = np.nan, np.nan

        return ra, dec

    _NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    def _pget_numbers(tool, param):
        """Numbers held in a tool's output parameter (comma-separated string)."""
        out = subprocess.run(
            ["pget", tool, param], capture_output=True, text=True
        ).stdout
        return [float(v) for v in re.findall(_NUM, out)]

    def _stdout_pair(text, label):
        """Parse a 'label: ( a b )' line from dmstat's screen output."""
        m = re.search(rf"{label}\s*:\s*\(?\s*({_NUM})[\s,]+({_NUM})", text)
        if m:
            return float(m.group(1)), float(m.group(2))
        return np.nan, np.nan

    def dmstat_centroid(image_path, region_file):

        cmd = [
            "dmstat",
            f"{image_path}[sky=region({region_file})]",
            "centroid=yes",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] dmstat centroid failed for {region_file}: {result.stderr}")
            return np.nan, np.nan

        vals = _pget_numbers("dmstat", "out_cntrd_phys")
        if len(vals) >= 2:
            return vals[0], vals[1]
        return _stdout_pair(result.stdout, r"cntrd\[phys\]")

    # 95% ellipse of a 2d gaussian: semi-axes are 2.448 sigma (chi2 = 5.991), not 2 sigma
    CONF_LEVEL = 0.95
    K_CONF = math.sqrt(-2.0 * math.log(1.0 - CONF_LEVEL))

    # jacobian step in sky pixels: big enough to beat dmcoords rounding,
    # small enough that the projection stays linear
    JAC_STEP_PIX = 50.0

    # fit regions are 80% ECF ellipses. treat the PSF as a gaussian truncated there
    # (R80 = 1.794 sigma), which gives sigma_axis ~ 0.431 * semi-axis
    PSF_ECF = 0.8
    _u80 = -math.log(1.0 - PSF_ECF)
    _trunc_var = (1.0 - (1.0 + _u80) * (1.0 - PSF_ECF)) / PSF_ECF
    SIGMA_PER_SEMIAXIS = math.sqrt(_trunc_var / (2.0 * _u80))

    def event_centroid_covariance(evt_file, region_file, elo_kev, ehi_kev):
        """(N, 2x2 covariance of the event mean) for events in the region. NaN for N < 3,
        shaky below ~20 so only used as a sanity check."""
        nan_cov = np.full((2, 2), np.nan)
        spec = (
            f"{evt_file}[energy={elo_kev * 1000.0:g}:{ehi_kev * 1000.0:g}]"
            f"[sky=region({region_file})][cols x,y]"
        )
        try:
            tab = crates_read_file(spec)
            x = np.asarray(tab.get_column("x").values, dtype=float)
            y = np.asarray(tab.get_column("y").values, dtype=float)
        except Exception as e:
            print(f"[WARN] could not read events for {region_file}: {e}")
            return np.nan, nan_cov

        n = len(x)
        if n < 3:
            return float(n), nan_cov
        return float(n), np.cov(np.vstack([x, y]), ddof=1) / n

    def read_region_ellipse(region_file):
        """(x, y, r1, r2, angle_deg) of the first ellipse in a region file, else None."""
        with open(region_file) as f:
            for line in f:
                m = re.search(r"ellipse\(([^)]+)\)", line, re.IGNORECASE)
                if m:
                    try:
                        parts = [float(p) for p in m.group(1).split(",")]
                    except ValueError:
                        return None
                    if len(parts) >= 5:
                        return tuple(parts[:5])
        return None

    def psf_region_covariance(region_file):
        """Per-event covariance (sky pix^2) of the truncated PSF, from the ellipse axes and angle (CCW from +x)."""
        ell = read_region_ellipse(region_file)
        if ell is None:
            print(f"[WARN] no ellipse found in {region_file}")
            return np.full((2, 2), np.nan)
        _, _, r1, r2, ang = ell
        th = math.radians(ang)
        rot = np.array([[math.cos(th), -math.sin(th)],
                        [math.sin(th),  math.cos(th)]])
        sig = SIGMA_PER_SEMIAXIS * np.array([r1, r2])
        return rot @ np.diag(sig ** 2) @ rot.T

    def sky_to_tangent_jacobian(x0, y0, ra0, dec0, step=JAC_STEP_PIX):
        """d(East, North)/d(x, y) in arcsec per sky pixel, by finite differences with dmcoords. None if dmcoords fails."""
        ra_x, dec_x = pix_to_radec(x0 + step, y0)
        ra_y, dec_y = pix_to_radec(x0, y0 + step)
        if not np.all(np.isfinite([ra0, dec0, ra_x, dec_x, ra_y, dec_y])):
            return None

        cos_dec = math.cos(math.radians(dec0))

        def offset_arcsec(ra, dec):
            d_ra = (ra - ra0 + 180.0) % 360.0 - 180.0      # handle ra wrap-around
            return np.array([d_ra * cos_dec, dec - dec0]) * 3600.0

        # columns are d(E,N)/dx and d(E,N)/dy
        return np.column_stack(
            [offset_arcsec(ra_x, dec_x) / step, offset_arcsec(ra_y, dec_y) / step]
        )

    def error_ellipse_95(cov_xy, jac):
        """95% error ellipse (semi-major, semi-minor in arcsec, PA in deg E of N) from a sky-pixel covariance.
        PA means nothing when the axes are nearly equal."""
        if jac is None or not np.all(np.isfinite(cov_xy)):
            return np.nan, np.nan, np.nan

        cov_en = jac @ cov_xy @ jac.T                      # arcsec^2, axes (E, N)
        w, v = np.linalg.eigh(cov_en)                      # ascending eigenvalues
        w = np.clip(w, 0.0, None)

        semi_minor = K_CONF * math.sqrt(w[0])
        semi_major = K_CONF * math.sqrt(w[1])
        # major-axis eigenvector is (East, North) = (v[0,1], v[1,1])
        pa = math.degrees(math.atan2(v[0, 1], v[1, 1])) % 180.0
        return semi_major, semi_minor, pa

    _psf_note = {"done": False}

    def psf_centroid(psf_file):
    
        if psf_center_coords == "physical":
            param, label = "out_cntrd_phys", r"cntrd\[phys\]"
        elif psf_center_coords == "logical":
            param, label = "out_cntrd_log", r"cntrd\[log\]"
        else:
            raise ValueError(
                f"psf_center_coords must be 'physical' or 'logical', got {psf_center_coords!r}"
            )

        result = subprocess.run(
            ["dmstat", psf_file, "centroid=yes"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"dmstat failed on {psf_file}: {result.stderr}")

        vals = _pget_numbers("dmstat", param)
        if len(vals) >= 2:
            x, y = vals[0], vals[1]
        else:
            x, y = _stdout_pair(result.stdout, label)
        if not (np.isfinite(x) and np.isfinite(y)):
            raise RuntimeError(
                f"could not read {label} from dmstat for {psf_file}:\n{result.stdout}"
            )

        if psf_center_coords == "physical" and not _psf_note["done"]:
            _psf_note["done"] = True
            nx, ny = get_image_dimensions(psf_file)
            if not (0 <= x <= nx and 0 <= y <= ny):
                print(
                    f"[NOTE] psf.center = ({x:.2f}, {y:.2f}) [physical] lies outside the "
                    f"{nx}x{ny} PSF kernel array. Sherpa's PSFModel.center defaults to "
                    f"({int(nx / 2)}, {int(ny / 2)}) here (array pixels). If MLE_RA/DEC "
                    f"disagree with Centroid_RA/DEC and Wav_RA/DEC, set "
                    f"psf_center_coords = 'logical'."
                )
        return x, y

    # stage 1: background fit
    for i in range(1, number_sources + 1):

        image = srcless_tpl.format(i=i)
        bkgmap = bkgmap_tpl.format(i=i) 
        emap = emap_tpl.format(i=i) 
        region_name = region_bkg_tpl.format(i=i)

        cmd = ["dmstat", f"{image}[sky=region({region_name})]", "centroid=no"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        sum_match = re.search(r"sum:\s+([0-9eE+.-]+)", result.stdout)
        good_match = re.search(r"good:\s+([0-9eE+.-]+)", result.stdout)

        if sum_match and good_match:
            sum_val = float(sum_match.group(1))
            good_val = float(good_match.group(1))
            if good_val > 0:
                bkg_count_rates.append(sum_val / good_val)
            else:
                # no good pixels, rate undefined
                bkg_count_rates.append(np.nan)
        else:
            bkg_count_rates.append(np.nan)

        load_image(i, image)

        bkg_id = f"bkg{i}"
        load_table_model(bkg_id, bkgmap)
        bkg_model = get_model_component(bkg_id)
        bkg_model.ampl.min = 0

        emap_id = f"emap{i}"
        load_table_model(emap_id, emap)
        freeze(emap_id)

        set_coord("physical")
        notice2d_id(i, region_name)
        set_model(i, bkg_model)

    set_stat("cash")
    set_method("moncar")

    for i in range(1, number_sources + 1):
        try:
            fit(i)
            fitres = get_fit_results()

            ampl_value = 0
            for name, val in zip(fitres.parnames, fitres.parvals):
                if "ampl" in name:
                    ampl_value = val
                    break

            bkg_amps.append(ampl_value)

        except Exception as e:
            if "no noticed bins found" in str(e):
                bkg_amps.append(0)
            else:
                raise

    print("bkg_amps =", bkg_amps)
    print(get_fit_results())

    # stage 2: source position fit
    clean()

    for i in range(1, number_sources + 1):
        image = image_tpl.format(i=i)
        bkgmap = bkgmap_tpl.format(i=i)
        emap_file = emap_tpl.format(i=i)
        region_name = region_smallbkg_tpl.format(i=i)
        psf_file = psf_tpl.format(i=i)

        bin_size = bin_map.get(i, 1)

        load_image(i, image)

        bkg_id = f"bkg{i}"
        load_table_model(bkg_id, bkgmap)
        bkg_model = get_model_component(bkg_id)
        bkg_model.ampl = bkg_amps[i - 1]
        freeze(bkg_model.ampl)

        emap_id = f"emap{i}"
        load_table_model(emap_id, emap_file)

        # psf
        psf_id = f"psf{i}"
        load_psf(psf_id, psf_file)
        set_psf(i, psf_id)

        psfobj = get_psf(i)

        x_centroid, y_centroid = psf_centroid(psf_file)
        psfobj.center = [x_centroid, y_centroid]

        # emap scaling
        dmstat_result = subprocess.run(
            ["dmstat", emap_file, "centroid=no"], capture_output=True, text=True
        )

        area = None
        dmlist_result = subprocess.run(
            ["dmlist", emap_file, "blocks"], capture_output=True, text=True
        )

        for line in dmlist_result.stdout.splitlines():
            if "Image" in line and "(" in line:
                dims = line.split("(")[1].split(")")[0].split("x")
                area = int(dims[0]) * int(dims[1])

        sum_value = None
        sum_match = re.search(r"sum:\s+([0-9eE+.-]+)", dmstat_result.stdout)
        if sum_match:
            sum_value = float(sum_match.group(1))

        ratio = area / sum_value if area and sum_value else 1.0
        emap_model = get_model_component(emap_id)
        emap_model.ampl = ratio
        freeze(emap_model.ampl)

        set_coord("physical")
        notice2d_id(i, region_name)

        mdl_name = f"mdl{i}"
        create_model_component("sigmagauss2d", mdl_name)
        mdl = get_model_component(mdl_name)

        xpos, ypos = read_region_center(region_name)

        x_pts, y_pts = get_indep(i, filter=True)

        if len(x_pts) > 0:
            x_min, x_max = float(x_pts.min()), float(x_pts.max())
            y_min, y_max = float(y_pts.min()), float(y_pts.max())

            region_x_spans[i - 1] = x_max - x_min
            region_y_spans[i - 1] = y_max - y_min

            set_par(mdl.xpos, xpos, min=x_min, max=x_max)
            set_par(mdl.ypos, ypos, min=y_min, max=y_max)
        else:
            mdl.xpos, mdl.ypos = xpos, ypos

        mdl.sigma_a = mdl.sigma_b = 1

        # region_counts isn't filled until stage 4, so get the upper bound on
        # the amplitude for this region here
        cmd_cts = ["dmstat", f"{img_file}[sky=region({region_name})]", "centroid=no"]
        result_cts = subprocess.run(cmd_cts, capture_output=True, text=True)
        sum_match_cts = re.search(r"sum:\s+([0-9eE+.-]+)", result_cts.stdout)
        max_counts = float(sum_match_cts.group(1)) if sum_match_cts else np.inf

        set_par(mdl.ampl, min=0, max=max_counts)
        freeze(mdl.sigma_a)
        freeze(mdl.sigma_b)

        set_full_model(
            i, (eval(f"psf{i}(mdl)") * eval(emap_id) + bkg_model) * (bin_size)
        )

        set_stat("cash")
        set_method("neldermead")

        # fit the position and update the region center
        try:
            fit(i)

            mdl = get_model_component(f"mdl{i}")
            fitted_xpos, fitted_ypos = mdl.xpos.val, mdl.ypos.val

            update_region_center(
                region_src_tpl.format(i=i), fitted_xpos, fitted_ypos
            )

            ra_corr, dec_corr = pix_to_radec(fitted_xpos, fitted_ypos)
            corrected_ra[i - 1] = ra_corr
            corrected_dec[i - 1] = dec_corr

        except Exception as e:
            if "no noticed bins found" in str(e):
                pass  # keep the original center
            else:
                raise

    print("src_fit =", src_fit)
    print("dof =", dof)
    print("emap_region_vals =", emap_region_vals)
    print("mean_region_exposure =", mean_region_exposure)
    print("normalised_exposure =", normalised_exposure)

    # stage 2b: centroid on the raw image in the fitted region, plus a 95% error ellipse
    fit_raw_ra = np.full(number_sources, np.nan)
    fit_raw_dec = np.full(number_sources, np.nan)
    fit_raw_95_major = np.full(number_sources, np.nan)
    fit_raw_95_minor = np.full(number_sources, np.nan)
    fit_raw_95_pa = np.full(number_sources, np.nan)
    fit_raw_n_events = np.full(number_sources, np.nan)

    for i in range(1, number_sources + 1):
        idx = i - 1
        fit_region = region_src_tpl.format(i=i)

        print(f"Source {i}: centroid (fit region, raw image)")

        x_phys, y_phys = dmstat_centroid(img_file, fit_region)

        jac = None
        if np.isfinite(x_phys) and np.isfinite(y_phys):
            ra_c, dec_c = pix_to_radec(x_phys, y_phys)
            fit_raw_ra[idx], fit_raw_dec[idx] = ra_c, dec_c
            if np.isfinite(ra_c) and np.isfinite(dec_c):
                jac = sky_to_tangent_jacobian(x_phys, y_phys, ra_c, dec_c)

        n_ev, cov_emp = event_centroid_covariance(evt2_file, fit_region, elo, ehi)
        fit_raw_n_events[idx] = n_ev

        # centroid covariance from the PSF: Sigma_PSF / N
        cov_xy = np.full((2, 2), np.nan)
        if np.isfinite(n_ev) and n_ev >= 1:
            cov_xy = psf_region_covariance(fit_region) / n_ev

        (
            fit_raw_95_major[idx],
            fit_raw_95_minor[idx],
            fit_raw_95_pa[idx],
        ) = error_ellipse_95(cov_xy, jac)

        print(
            f"    centroid        RA={fit_raw_ra[idx]:.6f} DEC={fit_raw_dec[idx]:.6f}  "
            f"N_events={n_ev:.0f}  "
            f"95% ellipse: a={fit_raw_95_major[idx]:.3f} arcsec  "
            f"b={fit_raw_95_minor[idx]:.3f} arcsec  "
            f"PA={fit_raw_95_pa[idx]:.1f} deg (E of N)"
        )

        # check against the raw event scatter (bright sources only)
        if n_ev >= 20:
            a_e, b_e, pa_e = error_ellipse_95(cov_emp, jac)
            print(
                f"    (event-scatter check: a={a_e:.3f} b={b_e:.3f} "
                f"PA={pa_e:.1f})"
            )

    # stage 3: wavdetect position errors (2 sigma)

    with fits.open(srclist_fits) as hdul:
        src_table = hdul[1].data
        src_colnames = [c.upper() for c in hdul[1].columns.names]
        number_sources = len(src_table)
        # plain copies so nothing depends on the memory-mapped file
        ra_err_all = np.array(src_table["RA_ERR"], dtype=float)
        dec_err_all = np.array(src_table["DEC_ERR"], dtype=float)
        x_err_all = (
            np.array(src_table["X_ERR"], dtype=float)
            if "X_ERR" in src_colnames
            else np.full(number_sources, np.nan)
        )
        y_err_all = (
            np.array(src_table["Y_ERR"], dtype=float)
            if "Y_ERR" in src_colnames
            else np.full(number_sources, np.nan)
        )
        ra_err_unit = hdul[1].columns["RA_ERR"].unit

    WAV_ERR_TO_2SIG_ARCSEC = 3600.0 * ci_nsigma
    
    if str(ra_err_unit or "deg").lower().startswith("arcsec"):
        wav_err_to_arcsec = 1.0                  # RA_ERR already in arcsec
        pix_floor = arcsec_per_pix               # one pixel, in arcsec
    else:
        wav_err_to_arcsec = 3600.0               # deg -> arcsec
        pix_floor = arcsec_per_pix / 3600.0      # one pixel, in deg

    WAV_ERR_TO_2SIG_ARCSEC = wav_err_to_arcsec * ci_nsigma

    bad_ra = ~np.isfinite(ra_err_all) | (ra_err_all <= 0)
    bad_dec = ~np.isfinite(dec_err_all) | (dec_err_all <= 0)

    n_bad = int(np.sum(bad_ra | bad_dec))
    if n_bad:
        print(
            f"[WARN] {n_bad} of {number_sources} sources have RA_ERR/DEC_ERR <= 0 "
            f"or non-finite in {srclist_fits} (RA_ERR unit: {ra_err_unit}); "
            f"using a one-pixel floor for them:"
        )
        for k in np.flatnonzero(bad_ra | bad_dec):
            print(
                f"    source {k + 1}: RA_ERR={ra_err_all[k]:.3e} "
                f"DEC_ERR={dec_err_all[k]:.3e} "
                f"X_ERR={x_err_all[k]:.3e} Y_ERR={y_err_all[k]:.3e}"
            )

    wav_2sig_ra = np.where(bad_ra, pix_floor, ra_err_all) * WAV_ERR_TO_2SIG_ARCSEC
    wav_2sig_dec = np.where(bad_dec, pix_floor, dec_err_all) * WAV_ERR_TO_2SIG_ARCSEC

    # stage 4: background-only fit
    clean()

    for i in range(1, number_sources + 1):

        image = image_tpl.format(i=i)
        bkgmap = bkgmap_tpl.format(i=i)
        emap = emap_tpl.format(i=i)
        region_name = region_src_tpl.format(i=i)

        load_image(i, image)

        bkg_id = f"bkg{i}"
        load_table_model(bkg_id, bkgmap)
        bkg_model = get_model_component(bkg_id)

        set_par(
            bkg_model.ampl,
            bkg_amps[i - 1],
            min=bkg_amps[i - 1],
            max=bkg_amps[i - 1],
        )

        emap_id = f"emap{i}"
        load_table_model(emap_id, emap)
        freeze(emap_id)

        set_coord("physical")

        cmd = [
            "dmstat",
            f"{img_file}[sky=region({region_name})]",
            "centroid=no",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        sum_match = re.search(r"sum:\s+([0-9eE+.-]+)", result.stdout)
        good_match = re.search(r"good:\s+([0-9eE+.-]+)", result.stdout)

        if sum_match:
            region_counts.append(float(sum_match.group(1)))
        else:
            region_counts.append(np.nan)

        if good_match:
            region_pixels.append(float(good_match.group(1)))
        else:
            region_pixels.append(np.nan)

        notice2d_id(i, region_name)

        set_model(i, bkg_model)

    set_stat("cash")
    set_method("moncar")

    for i in range(1, number_sources + 1):
        try:
            fit(i)
            bkg_fit.append(get_fit_results().statval)
        except Exception as e:
            print(f"\nDataset {i} failed")
            print(type(e))
            print(e)
            raise
    print("region_counts =", region_counts)
    print("region_pixels =", region_pixels)
    print("bkg_fit =", bkg_fit)
            
    # stage 5: full model fit
    clean()

    for i in range(1, number_sources + 1):

        image = image_tpl.format(i=i)
        bkgmap = bkgmap_tpl.format(i=i)
        emap_file = emap_tpl.format(i=i)
        region_name = region_src_tpl.format(i=i)
        psf_file = psf_tpl.format(i=i)

        bin_size = bin_map.get(i, 1)

        load_image(i, image)

        bkg_id = f"bkg{i}"
        load_table_model(bkg_id, bkgmap)
        bkg_model = get_model_component(bkg_id)
        bkg_model.ampl = bkg_amps[i - 1]
        freeze(bkg_model.ampl)

        emap_id = f"emap{i}"
        load_table_model(emap_id, emap_file)

        # psf
        psf_id = f"psf{i}"
        load_psf(psf_id, psf_file)
        set_psf(i, psf_id)

        psfobj = get_psf(i)

        x_centroid, y_centroid = psf_centroid(psf_file)
        psfobj.center = [x_centroid, y_centroid]

        # emap scaling
        dmstat_result = subprocess.run(
            ["dmstat", emap_file, "centroid=no"], capture_output=True, text=True
        )

        area = None
        dmlist_result = subprocess.run(
            ["dmlist", emap_file, "blocks"], capture_output=True, text=True
        )

        for line in dmlist_result.stdout.splitlines():
            if "Image" in line and "(" in line:
                dims = line.split("(")[1].split(")")[0].split("x")
                area = int(dims[0]) * int(dims[1])

        sum_value = None
        mean_sum_value = None
        sum_match = re.search(r"sum:\s+([0-9eE+.-]+)", dmstat_result.stdout)
        mean_sum_match = re.search(r"mean:\s+([0-9eE+.-]+)", dmstat_result.stdout)
        if sum_match:
            sum_value = float(sum_match.group(1))
        if mean_sum_match:
            mean_sum_value = float(mean_sum_match.group(1))

        # mean exposure in the region
        dmstat_reg_result = subprocess.run(
            ["dmstat", f"{emap_file}[sky=region({region_name})]", "centroid=no"],
            capture_output=True,
            text=True,
        )

        mean_match = re.search(r"mean:\s+([0-9eE+.-]+)", dmstat_reg_result.stdout)
        if mean_match:
            mean_exp = float(mean_match.group(1))
            mean_region_exposure.append(mean_exp / bin_size**2)
        else:
            mean_exp = np.nan
            mean_region_exposure.append(np.nan)

        # normalised exposure
        if sum_value and not np.isnan(mean_exp):
            norm_exp = mean_exp / mean_sum_value
        else:
            norm_exp = np.nan
        normalised_exposure.append(norm_exp)

        ratio = area / sum_value if area and sum_value else 1.0
        emap_model = get_model_component(emap_id)
        emap_model.ampl = ratio
        freeze(emap_model.ampl)

        emap_region_vals.append(ratio)

        set_coord("physical")
        notice2d_id(i, region_name)

        mdl_name = f"mdl{i}"
        create_model_component("sigmagauss2d", mdl_name)
        mdl = get_model_component(mdl_name)

        xpos, ypos = read_region_center(region_name)
        mdl.xpos, mdl.ypos = xpos, ypos
        freeze(mdl.xpos)
        freeze(mdl.ypos)

        mdl.sigma_a = mdl.sigma_b = 1
        set_par(mdl.ampl, min=0, max=region_counts[i - 1])
        freeze(mdl.sigma_a)
        freeze(mdl.sigma_b)

        set_full_model(
            i, (eval(f"psf{i}(mdl)") * eval(emap_id) + bkg_model) * (bin_size)
        )

    set_stat("cash")
    set_method("neldermead")

    for i in range(1, number_sources + 1):
        try:
            fit(i)
            res = get_fit_results()
            src_fit.append(res.statval)
            dof.append(res.dof)
        except Exception as e:
            if "no noticed bins found" in str(e):
                src_fit.append(0)
                dof.append(0)
            else:
                raise

    print("src_fit =", src_fit)
    print("dof =", dof)
    print("emap_region_vals =", emap_region_vals)
    print("mean_region_exposure =", mean_region_exposure)
    print("normalised_exposure =", normalised_exposure)

    # stage 6: likelihood and classification

    results = []
    classifications = []

    for dof_val, cnull, cbest, count, pix, exp_norm in zip(
        dof, bkg_fit, src_fit, region_counts, region_pixels, normalised_exposure
    ):

        s = dof_val * 0.5
        x = (cnull - cbest) * 0.5

        val = -log_gamma_ratio(s, x)

        nan_flag = False

        if np.isinf(val):
            val = 1e3
        elif np.isnan(val):
            val = 0.1
            nan_flag = True

        results.append(val)

        exposure = val

        # threshold: the larger of 2.5 and the expected background counts
        count_thresh = max(2.5, global_bkg_count_rate * pix * exp_norm)
        count_thresholds.append(count_thresh)

        if count < count_thresh:
            classifications.append("Too Faint")
        elif nan_flag:
            classifications.append("nan")
        elif exposure > 10:
            classifications.append("True")
        elif exposure > 2.5:
            classifications.append("Marginal")
        else:
            classifications.append("Unlikely")

    # wavdetect positions
    with fits.open(srclist_fits) as hdul:
        src_table = hdul[1].data
    ra1 = src_table["RA"]
    dec1 = src_table["DEC"]

    # stage 6b: photon flux and 2 sigma interval (bayesian, flat prior on S >= 0)
    
    PSF_ENCLOSED_FRAC = 0.8

    # percentiles for the 2 sigma (~95%) interval
    SIGMA2_LOW, SIGMA2_HIGH = 2.5, 97.5

    def source_counts_quantile(p, N, B):
        """Source counts S where posterior CDF F(S) = p."""
        q_B = gammaincc(N + 1.0, B)
        if not np.isfinite(q_B) or q_B <= 0:      # underflow guard (B >> N)
            return np.nan
        s_plus_b = gammainccinv(N + 1.0, (1.0 - p) * q_B)
        if not np.isfinite(s_plus_b):
            return np.nan
        return max(s_plus_b - B, 0.0)

    p_lo = SIGMA2_LOW / 100.0
    p_hi = SIGMA2_HIGH / 100.0

    photon_flux = np.full(number_sources, np.nan)
    photon_flux_2sig_minus = np.full(number_sources, np.nan)
    photon_flux_2sig_plus = np.full(number_sources, np.nan)

    for isrc, (N_obs, pix_src, exp_norm_src, exp_cm2s) in enumerate(
        zip(region_counts, region_pixels, normalised_exposure, mean_region_exposure)
    ):
        B_src = global_bkg_count_rate * pix_src * exp_norm_src   # mean background counts

        if not np.all(np.isfinite([N_obs, B_src, exp_cm2s])) or exp_cm2s <= 0:
            continue

        S_best = max(N_obs - B_src, 0.0)
        S_lo = source_counts_quantile(p_lo, N_obs, B_src)
        S_hi = source_counts_quantile(p_hi, N_obs, B_src)

        conv = 1.0 / (exp_cm2s * PSF_ENCLOSED_FRAC)   # counts -> photons/cm^2/s

        photon_flux[isrc] = S_best * conv
        if np.isfinite(S_lo):
            photon_flux_2sig_minus[isrc] = max(S_best - S_lo, 0.0) * conv
        if np.isfinite(S_hi):
            photon_flux_2sig_plus[isrc] = (S_hi - S_best) * conv

    print("photon_flux =", photon_flux)


    # stage 6c: nh1 (HI4PI) and nh2 (CO) at each source
    nh1_values = np.full(number_sources, np.nan)
    nh2_values = np.full(number_sources, np.nan)

    if not NH_LOOKUP_AVAILABLE:
        print(
            "[INFO] nh_lookup.py not available: skipping the nh1 / W_CO / nh2 "
            "lookups (nh1 and nh2 columns will be NaN)."
        )
    else:
        if NH_POSITION == "wav":
            nh_ra, nh_dec = np.asarray(ra1, dtype=float), np.asarray(dec1, dtype=float)
        else:
            nh_ra, nh_dec = fit_raw_ra, fit_raw_dec

        for idx in range(number_sources):
            # an nH problem should never cost us the ObsID
            try:
                est = get_nh_estimates(
                    nh_ra[idx], nh_dec[idx], nh_pfile_dir, WCO_MAP, xco=XCO
                )
            except Exception as e:
                print(f"[WARN] nh lookup failed for source {idx + 1} ({e}); nh1/nh2 left as NaN")
                continue
            nh1_values[idx] = est["nh1"]
            nh2_values[idx] = est["nh2"]
            print(
                f"Source {idx + 1}: nh1 = {est['nh1']:.3E}  "
                f"W_CO = {est['wco']:.2f} K km/s  nh2 = {est['nh2']:.3E} cm^-2"
            )

    # stage 6d: off-axis angle theta, at the wavdetect position (never NaN)
    theta_values = np.full(number_sources, np.nan)

    for idx in range(number_sources):
        theta_values[idx] = get_theta(
            evt2_file, asol_file, float(ra1[idx]), float(dec1[idx])
        )
        print(f"Source {idx + 1}: theta = {theta_values[idx]:.3f} arcmin")

    # write csv

    n_out = len(results)

    df_out = pd.DataFrame(
        {
            "region_id": np.arange(1, len(results) + 1),
            "Wav_RA": ra1[: len(results)],
            "Wav_DEC": dec1[: len(results)],
            "MLE_RA": corrected_ra[: len(results)],
            "MLE_DEC": corrected_dec[: len(results)],
            "Centroid_RA": fit_raw_ra[:n_out],
            "Centroid_DEC": fit_raw_dec[:n_out],
            "Wav_2sigma_RA": wav_2sig_ra[: len(results)],
            "Wav_2sigma_DEC": wav_2sig_dec[: len(results)],
            "Centroid_95_semimajor": fit_raw_95_major[:n_out],   # arcsec
            "Centroid_95_semiminor": fit_raw_95_minor[:n_out],   # arcsec
            "Centroid_95_PA": fit_raw_95_pa[:n_out],             # deg, East of North, 0-180
            "Centroid_N_events": fit_raw_n_events[:n_out],
            "theta": theta_values[:n_out],   # off-axis angle, arcmin
            "region_counts": region_counts,
            "region_pixels/psf_size": region_pixels,
            "mean_region_exposure": mean_region_exposure,
            "normalised_exposure": normalised_exposure,
            "count_threshold": count_thresholds,
            "bkg_count_density": bkg_count_rates,
            "likelihood": results,
            "source_classification": classifications,
            "nh1": nh1_values[:n_out],                        # HI, cm^-2 (HI4PI)
            "nh2": nh2_values[:n_out],                        # H2 molecules, cm^-2
            "photon_flux": photon_flux,                       # ph/cm^2/s
            "photon_flux_2sig_minus": photon_flux_2sig_minus, # ph/cm^2/s
            "photon_flux_2sig_plus": photon_flux_2sig_plus,   # ph/cm^2/s
        }
    )

    df_out.to_csv(output_csv_path, index=False)

    print(f"[INFO] Final CSV created at: {output_csv_path}")
    print(df_out)
    
if __name__ == "__main__":

    for obsid in obsids:
        print("=" * 60)
        print(f"Starting processing for OBSID {obsid}")
        print("=" * 60)

        try:
            run_obsid(obsid)

        except Exception as e:
            print(f"ERROR processing OBSID {obsid}: {e}")
            print("Continuing to next OBSID...\n")
            continue

        print(f"Finished OBSID {obsid}\n")
