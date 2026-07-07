# ==========================================================
# IMPORTS
# ==========================================================
import subprocess
import os
from pathlib import Path
from astropy.io import fits
import re
import csv
import math
import glob
import tempfile
import pandas as pd
import shutil
from sherpa.astro.ui import *
import numpy as np
from scipy.special import gammaincc, gammaln, exp1

# ==========================================================
# INPUTs
# ==========================================================

# ----------------------------
# USER INPUTS
# ----------------------------

obsid = "29265" #<---------- JUST SET THESE
base_dir = "ponti_research" #<---------- JUST SET THESE

# ----------------------------
# Adjustable INPUTS
# ----------------------------

# ----------------------------
# Dont touch paths and directories
# ----------------------------

bin_size_scale = 0.5 #sets how much to rebin (originally)
smoothing_scale = 50 # Gaussian smoothing scale (pixels)

# ----------------------------
# script 1 inputs
# ----------------------------

input_dir = f"{base_dir}/{obsid}"
repro_dir = f"{input_dir}/repro"

evt2_file = f"{repro_dir}/acisf{obsid}_repro_evt2.fits"

# Outputs from merge_obs
img_file = f"{repro_dir}/{obsid}_broad_thresh.img"
expmap_file = f"{repro_dir}/{obsid}_broad_thresh.expmap"
psfmap_file = f"{repro_dir}/{obsid}_broad_thresh.psfmap"

# wavdetect outputs
srclist_fits = f"{repro_dir}/{obsid}_sourcelist.fits"
regfile = f"{repro_dir}/{obsid}_wavdetect.reg"

# ----------------------------
# script 3 inputs
# ----------------------------

input_region_file = regfile

psf_image_template = (
    f"{repro_dir}/{obsid}_PSF_{{region}}.psf"
)

psf_fit_dir = f"{repro_dir}/psf_fit_regions"
psf_bkg_dir = f"{repro_dir}/psf_bkg_regions"

bin_size_path = os.path.join(repro_dir, f"{obsid}_binsizes.csv")

combined_psf_path = os.path.join(f"{psf_bkg_dir}/all_psf_regions.reg")

# ----------------------------
# script 4 inputs
# ----------------------------
image_file = img_file
all_region_file =f"{psf_bkg_dir}/all_psf_regions.reg"
srclist = f"{psf_bkg_dir}/psf_smallbkg_regions.lis"
bkglist = f"{psf_bkg_dir}/psf_bigbkg_regions.lis"
background_image = f"{repro_dir}/{obsid}_background.img"

# ----------------------------
# script 5 inputs
# ----------------------------
input_image = img_file
input_background = background_image
input_emap = expmap_file
input_psf_template = f"{repro_dir}/{obsid}_PSF_{{region}}.psf"
bin_sizes_csv = bin_size_path

# ----------------------------
# script 6 inputs
# ----------------------------

image_tpl = f"{repro_dir}/image_rebin_{{i}}.fits"
bkgmap_tpl = f"{repro_dir}/bkg_rebin_{{i}}.fits"
emap_tpl = f"{repro_dir}/emap_rebin_{{i}}.fits"

region_bkg_tpl = f"{psf_bkg_dir}/psf_bigbkg_{{i:03d}}.reg"
region_src_tpl = f"{psf_fit_dir}/psf_fit_{{i:03d}}.reg"

psf_tpl = f"{repro_dir}/psf_rebin_{{i}}.fits"

bin_csv_path = bin_size_path
output_csv_path = f"{repro_dir}/{obsid}_likelihood_results.csv"


# ==========================================================
# SCRIPT 1: PREPROCESSING AND WAVDETECT
# ==========================================================

# ----------------------------
# make directories
# ----------------------------

Path(base_dir).mkdir(parents=True, exist_ok=True)

# ----------------------------
# download Obsid
# ----------------------------

if not Path(input_dir).exists():
    print(f"Downloading Chandra ObsID {obsid}...")
    subprocess.run(
        ["download_chandra_obsid", obsid],
        cwd=base_dir,
        check=True
    )
    print("Download complete.\n")
else:
    print(f"ObsID {obsid} already exists. Skipping download.\n")

# ----------------------------
# Run chandra_repro
# ----------------------------

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
            "bands=broad",
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

# ----------------------------
# Run Wavdetect
# ----------------------------
if Path(srclist_fits).exists() and Path(regfile).exists():
    print(f"Wavdetect output already exists: {srclist_fits}, {regfile}")
    print("Skipping wavdetect.\n")
else:
    wavdetect_cmd = [
        "wavdetect",
        f"infile={img_file}",
        f"outfile={srclist_fits}",
        "scellfile=scell.fits",
        "imagefile=img.fits",
        "defnbkgfile=defn.fits",
        f"expfile={expmap_file}",
        f"psffile={psfmap_file}",
        f"regfile={regfile}",
        "scales=1.414 2 2.828 4 5.657 8 11.314 16 22.627 32 45.255 64",
        "sigthresh=1e-5",
        "clobber=yes"
    ]

    print("\nRunning wavdetect:")
    print(" ".join(wavdetect_cmd))
    subprocess.run(wavdetect_cmd, check=True)

    # ----------------------------
    # Cleanup files
    # ----------------------------

    temp_files = ["scell.fits", "img.fits", "defn.fits"]

    for f in temp_files:
        if Path(f).exists():
            os.remove(f)
            print(f"Deleted {f}")

print("\nPreProcessing complete!")

# ==========================================================
# SCRIPT 2: PSF 
# ==========================================================

# ----------------------------
# Read source list
# ----------------------------
with fits.open(srclist_fits) as hdul:
    data = hdul[1].data

# ----------------------------
# Run simulate_psf for each source
# ----------------------------
for region_number, row in enumerate(data, start=1):

    ra = float(row["RA"])
    dec = float(row["Dec"])      # Change to "DEC" if your column is uppercase

    outfile = f"{repro_dir}/{obsid}_PSF_{region_number}"

    cmd = [
        "simulate_psf",
        f"infile={evt2_file}",
        f"outroot={outfile}",
        f"ra={ra}",
        f"dec={dec}",
        "spectrum=none",
        "monoenergy=2.3",
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
# SCRIPT 3: REGIONS 
# ==========================================================

os.makedirs(psf_fit_dir, exist_ok=True)
os.makedirs(psf_bkg_dir, exist_ok=True)

# Regex pattern for ellipse regions
pattern = re.compile(r'ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)')

# Initialize lists to store bin sizes and PSF table rows
bin_sizes = []

# Store all PSF ellipse definitions for a combined region file
combined_psf_regions = []


with open(input_region_file, "r") as infile:
    count = 1

    for line in infile:
        match = pattern.search(line)

        if match:
            # Extract parameters
            a = match.group(1).strip()
            b = match.group(2).strip()

            #PSF region
            with tempfile.TemporaryDirectory() as tmpdir:

                # Input PSF image
                PSF_file = psf_image_template.format(region=count)

                # Temporary FITS files
                psf_fit_fits = os.path.join(tmpdir, "fit.fits")
                psf_bkg_fits = os.path.join(tmpdir, "bkg.fits")

                # Permanent region files
                psf_fit_reg = os.path.join(
                    psf_fit_dir,
                    f"psf_fit_{count:03d}.reg"
                )

                psf_bkg_reg = os.path.join(
                    psf_bkg_dir,
                    f"psf_bkg_{count:03d}.reg"
                )
                print(f"Using PSF file: {PSF_file}")

                # Run dmellipse
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

                # Convert to region file (CIAO dmmakereg expects region(<file>) literally).
                # The parentheses must be quoted or bash will try to interpret them.
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

                # Read the ASCII region file produced by dmmakereg, not the
                # binary FITS file produced by dmellipse.
                with open(psf_fit_reg, "r") as psf_infile:

                    found = False

                    for psf_line in psf_infile:

                        search = psf_pattern.search(psf_line)

                        if search:

                            found = True

                            # Extract PSF ellipse parameters
                            f = float(search.group(3))
                            g = float(search.group(4))
                            h = search.group(5).strip()

                            # Calculate adaptive bin size
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

                            # -------------------------
                            # PSF fit regions
                            # -------------------------
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

                # Run dmellipse
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

                # Convert to region file (quote to protect parentheses from bash)
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

                # Read the ASCII region file produced by dmmakereg, not the
                # binary FITS file produced by dmellipse.
                with open(psf_bkg_reg, "r") as psf_infile:

                    found = False

                    for psf_line in psf_infile:

                        search = psf_pattern.search(psf_line)

                        if search:

                            found = True

                            # Extract PSF ellipse parameters
                            i = float(search.group(3))
                            j = float(search.group(4))
                            k = search.group(5).strip()

                            # -------------------------
                            # notfit psf
                            # -------------------------
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

                            # Save for combined PSF region file
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

                            # -------------------------
                            # PSF background
                            # -------------------------
                            i_small = 1.1 * i
                            j_small = 1.1 * j

                            if i < 10 or j < 10:
                                i_big, j_big = 50, 50
                            else:
                                i_big, j_big = i * 5, j * 5

                            psf_bkg_text = (
                                "# Region file format: DS9 version 4.1\n"
                                "physical\n"
                                f"ellipse({a},{b},{i_big},{j_big},{k}) & "
                                f"!ellipse({a},{b},{i_small},{j_small},{k})\n"
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
                                f"ellipse({a},{b},{i_big},{j_big},{k}) & "
                                f"!ellipse({a},{b},{i_small},{j_small},{k})"
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

# -------------------------
# save binsize csv
# -------------------------

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

# -------------------------
# save combined PSF region
# -------------------------
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

# ==========================================================
# SCRIPT 4: BACKGROUND 
# ==========================================================
def run_cmd(cmd):
    """Run a shell command and stop if it fails."""
    print(f"\nRunning:\n{cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def main():

    # Create unique temporary filenames
    temp_copy = tempfile.NamedTemporaryFile(
        suffix=".fits", delete=False
    ).name

    temp_dmfilth = tempfile.NamedTemporaryFile(
        suffix=".fits", delete=False
    ).name

    try:

        # --------------------------------------------------
        # Step 1: Remove source regions
        # --------------------------------------------------
        cmd = (
            f'dmcopy "{image_file}[exclude sky=region({all_region_file})]" '
            f'outfile={temp_copy} clobber=yes'
        )
        run_cmd(cmd)

        # --------------------------------------------------
        # Step 2: Fill removed regions with Poisson noise
        # --------------------------------------------------
        cmd = (
            f"dmfilth infile={temp_copy} "
            f"outfile={temp_dmfilth} "
            f"method=POISSON "
            f"srclist=@{srclist} "
            f"bkglist=@{bkglist} "
            f"clobber=yes"
        )
        run_cmd(cmd)

        # --------------------------------------------------
        # Step 3: Smooth the background image
        # --------------------------------------------------
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
        # --------------------------------------------------
        # Remove temporary files
        # --------------------------------------------------
        for f in [temp_copy, temp_dmfilth]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Deleted temporary file: {f}")

if __name__ == "__main__":
    main()
    
# ==========================================================
# SCRIPT 5: REBINNING 
# ==========================================================

# Load bin sizes from CSV
df = pd.read_csv(bin_sizes_csv)
region_numbers = df['Region Number'].tolist()
bin_sizes = df['Bin Size'].tolist()

# Function to get image dimensions using dmlist
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

# Get dimensions for main image (used for image, background, emap)
main_xmax, main_ymax = get_image_dimensions(input_image)

# Cache for already processed bin sizes (for main files)
processed_bins = {}  # key: bin_size, value: dict of file outputs {image, bkg, emap}

# Loop over sources
for region, bin_size in zip(region_numbers, bin_sizes):
    bin_size_int = int(round(bin_size))

    # Prepare output names for this region
    region_outputs = {
        'image': f'{repro_dir}/image_rebin_{region}.fits',
        'bkg': f'{repro_dir}/bkg_rebin_{region}.fits',
        'emap': f'{repro_dir}/emap_rebin_{region}.fits'
    }
    
    psf_outfile = f'{repro_dir}/psf_rebin_{region}.fits'

    # Skip region if all outputs already exist
    all_outputs = list(region_outputs.values()) + [psf_outfile]

    if all(os.path.exists(f) for f in all_outputs):
        print(f"[SKIP] Region {region}: all rebinned files already exist.")
        continue
    # Handle main files
    if bin_size_int == 1:
        # Copy originals (no rebin)
        for infile, outfile in zip([input_image, input_background, input_emap],
                                   region_outputs.values()):
            print(f"[COPY] Bin size = 1 for region {region}, copying {infile} -> {outfile}")
            shutil.copy(infile, outfile)
    else:
        # Check if this bin size was processed before
        if bin_size_int in processed_bins:
            # Copy from cached outputs
            print(f"[CACHE] Bin size {bin_size_int} already processed. Copying for region {region}.")
            for key, cached_file in processed_bins[bin_size_int].items():
                shutil.copy(cached_file, region_outputs[key])
        else:
            # Process new bin size
            print(f"[REGRID] Bin size {bin_size_int} new. Rebinning main files for region {region}.")
            bin_expr = f"1:{main_xmax}:{bin_size},1:{main_ymax}:{bin_size}"

            # Create base outputs for this bin size (cache)
            cache_outputs = {
                'image': f'{repro_dir}/image_bin_{bin_size_int}.fits',
                'bkg': f'{repro_dir}/bkg_bin_{bin_size_int}.fits',
                'emap': f'{repro_dir}/emap_bin_{bin_size_int}.fits'
            }

            main_files = [
                (input_image, cache_outputs['image']),
                (input_background, cache_outputs['bkg']),
                (input_emap, cache_outputs['emap'])
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

            # Save cache
            processed_bins[bin_size_int] = cache_outputs

            # Copy to region outputs
            for key, cached_file in cache_outputs.items():
                shutil.copy(cached_file, region_outputs[key])

    # Handle PSF (unique per region)
    psf_infile = input_psf_template.format(region=region)
    psf_outfile = f'{repro_dir}/psf_rebin_{region}.fits'

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
# SCRIPT 6: MLE FITTING
# ==========================================================

#clean sherpa session
clean()

# -------------------------
# storage arrays
# -------------------------
bkg_amps = []
region_counts = []
bkg_fit = []
emap_region_vals = []
src_fit = []
dof = []
with fits.open(srclist_fits) as hdul:
    number_sources = len(hdul[1].data)


# -------------------------
# bin map
# -------------------------
bin_map = {}
with open(bin_csv_path, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        bin_map[int(row[0])] = int(row[1])

# -------------------------
# helper functions
# -------------------------
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


# -------------------------
# stage 1 background fit
# -------------------------
for i in range(1, number_sources + 1):

    image = image_tpl.format(i=i)
    bkgmap = bkgmap_tpl.format(i=i)
    emap = emap_tpl.format(i=i)
    region_name = region_bkg_tpl.format(i=i)

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

# -------------------------
# stage 2 background model fit
# -------------------------
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
        bkg_amps[i-1],
        min=bkg_amps[i-1],
        max=bkg_amps[i-1]
    )

    emap_id = f"emap{i}"
    load_table_model(emap_id, emap)
    freeze(emap_id)

    set_coord("physical")
    
    cmd = [
        "dmstat",
        f"{image}[sky=region({region_name})]",
        "centroid=no"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    match = re.search(r"sum:\s+([0-9eE+.-]+)", result.stdout)

    if match:
        region_counts.append(float(match.group(1)))
    else:
        region_counts.append(np.nan)

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
print("bkg_fit =", bkg_fit)


# -------------------------
# stage 3 full model fit 
# -------------------------
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

    # PSF
    psf_id = f"psf{i}"
    load_psf(psf_id, psf_file)
    set_psf(i, psf_id)

    psfobj = get_psf(i)

    result = subprocess.run(
        ["dmstat", psf_file, "centroid=yes"],
        capture_output=True, text=True
    )

    x_centroid = y_centroid = None
    for line in result.stdout.splitlines():
        if "cntrd" in line:
            m = re.search(r"\(\s*([0-9.+-eE]+)\s+([0-9.+-eE]+)\s*\)", line)
            if m:
                x_centroid, y_centroid = float(m.group(1)), float(m.group(2))

    psfobj.center = [x_centroid, y_centroid]

    # EMAP scaling
    dmstat_result = subprocess.run(
        ["dmstat", emap_file, "centroid=no"],
        capture_output=True, text=True
    )

    area = None
    dmlist_result = subprocess.run(["dmlist", emap_file, "blocks"],
                                   capture_output=True, text=True)

    for line in dmlist_result.stdout.splitlines():
        if "Image" in line and "(" in line:
            dims = line.split("(")[1].split(")")[0].split("x")
            area = int(dims[0]) * int(dims[1])

    sum_value = None
    for line in dmstat_result.stdout.splitlines():
        if "sum:" in line:
            sum_value = float(line.split()[1])

    ratio = area / sum_value if area and sum_value else 1.0
    emap_model = get_model_component(emap_id)
    emap_model.ampl = ratio
    freeze(emap_model.ampl)

    emap_region_vals.append(ratio / (bin_size ** 2))

    set_coord("physical")
    notice2d_id(i, region_name)

    mdl_name = f"mdl{i}"
    create_model_component("sigmagauss2d", mdl_name)
    mdl = get_model_component(mdl_name)

    xpos, ypos = read_region_center(region_name)
    mdl.xpos, mdl.ypos = xpos, ypos
    mdl.sigma_a = mdl.sigma_b = 1
    freeze(mdl.sigma_a)
    freeze(mdl.sigma_b)

    set_full_model(i, eval(f"psf{i}(mdl)") * eval(emap_id) + bkg_model)

set_stat("cash")
set_method("levmar")

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

# -------------------------
# stage 4 likelihood and classification
# -------------------------

results = []
classifications = []

for dof_val, cnull, cbest, emap_val, count in zip(
    dof, bkg_fit, src_fit, emap_region_vals, region_counts
):

    s = dof_val * 0.5
    x = (cnull - cbest) * 0.5

    val = -log_gamma_ratio(s, x)

    nan_flag = False

    if np.isinf(val):        # catches both +inf and -inf
        val = 1e3
    elif np.isnan(val):
        val = 0.1
        nan_flag = True

    results.append(val)

    exposure = val # * (1e-6 / emap_val) Remove the hash when you figure out exposure correction

    if count < 3.1:
        classifications.append("Source too small")
    elif nan_flag:
        classifications.append("nan")
    elif exposure > 10:
        classifications.append("True")
    elif exposure > 2:
        classifications.append("Maybe")
    else:
        classifications.append("False")

# -------------------------
# creating likelihood csv
# -------------------------
df_out = pd.DataFrame({
    "region_id": np.arange(1, len(results) + 1),
    "dof": dof,
    "bkg_fit_stat": bkg_fit,
    "src_fit_stat": src_fit,
    "emap_correction": emap_region_vals,
    "region_counts": region_counts,
    "likelihood": results,
    "source_classification": classifications
})

df_out.to_csv(output_csv_path, index=False)

print(f"[INFO] New CSV created at: {output_csv_path}")
print(df_out)