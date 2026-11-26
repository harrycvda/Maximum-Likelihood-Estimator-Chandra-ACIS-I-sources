"""
Optional nH estimates for wavlex_v1.py.
  nh1: atomic H from the HEASARC `nh` tool (closest HI4PI pixel)
  nh2: molecular H from the CO map (Dame et al. 2001) times X_CO
Needs HEASOFT (for `nh`) and wco_map.fits. If either is missing the matching
value is NaN and you get a warning, never an error. Also runs on its own:
  python nh_lookup.py RA DEC --wco wco_map.fits
"""

import os
import re
import subprocess

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

# ==========================================================
# SETTINGS
# ==========================================================
HEADAS_DIR = "/Users/Shared/heasoft/heasoft-6.36/aarch64-apple-darwin24.6.0"  # your HEASOFT install dir, change this
XCO = 0.51e20            # cm^-2 / (K km/s)


# ======================================================================
# nh1: HEASARC `nh` ftool, closest map pixel
# ======================================================================

# one row of nh's table: RA DEC Dist(deg) nH(cm^-2)
_ROW_RE = re.compile(
    r"^\s*([0-9]+\.[0-9]+)\s+([-+]?[0-9]+\.[0-9]+)\s+"   # RA, DEC of map pixel
    r"([0-9]+\.[0-9]+)\s+"                                # Dist(deg)
    r"([0-9]+\.?[0-9]*[eE][-+]?[0-9]+)\s*$"               # nH (cm^-2)
)


def parse_nh_output(text):
    """Return a list of (ra, dec, dist_deg, nh) tuples from nh's screen output."""
    rows = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m:
            rows.append(tuple(float(g) for g in m.groups()))
    return rows


def get_nh_closest(
    ra_deg,
    dec_deg,
    pfile_dir,
    equinox=2000,
    headas_dir=HEADAS_DIR,
    timeout=60,
):
    """
    nH [cm^-2] and distance [deg] of the closest map pixel, from the `nh` tool.
    (nan, nan) if the position isn't finite or nh fails.
    """
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return np.nan, np.nan

    os.makedirs(pfile_dir, exist_ok=True)

    # all three params on the command line so nh never prompts,
    # stdin is closed so a stray prompt fails instead of hanging
    script = (
        f'export HEADAS="{headas_dir}"; '
        f'source "$HEADAS/headas-init.sh" >/dev/null 2>&1; '
        f'export PFILES="{pfile_dir};$HEADAS/syspfiles"; '
        f"nh equinox={equinox} ra={ra_deg:.7f} dec={dec_deg:.7f}"
    )
    clean_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TERM": "dumb",
    }

    try:
        result = subprocess.run(
            ["bash", "-c", script],
            env=clean_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[WARN] nh timed out at RA={ra_deg:.6f} DEC={dec_deg:.6f}")
        return np.nan, np.nan

    rows = parse_nh_output(result.stdout)
    if result.returncode != 0 or not rows:
        print(
            f"[WARN] nh returned no usable table at RA={ra_deg:.6f} "
            f"DEC={dec_deg:.6f} (rc={result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()[-300:]}"
        )
        return np.nan, np.nan

    _, _, dist, nh = min(rows, key=lambda r: r[2])   # smallest Dist(deg)
    return nh, dist


# ======================================================================
# nh2: CO map (HEALPix, NESTED, Galactic) x X_CO
# ======================================================================

_map_cache = {}


def _spread_bits(v):
    """Put the bits of v into the even bit positions (0, 2, 4, ...)."""
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for bit in range(16):
        out |= ((v >> bit) & 1) << (2 * bit)
    return out


def ang2pix_nest(nside, lon_deg, lat_deg):
    """HEALPix NESTED pixel index containing (lon, lat) given in degrees."""
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))

    z = np.sin(np.radians(lat))                    # cos(colatitude)
    za = np.abs(z)
    tt = np.mod(np.radians(lon), 2 * np.pi) / (np.pi / 2)    # in [0, 4)

    # equatorial zone
    t1 = nside * (0.5 + tt)
    t2 = nside * z * 0.75
    jp_e = (t1 - t2).astype(np.int64)
    jm_e = (t1 + t2).astype(np.int64)
    ifp = jp_e // nside
    ifm = jm_e // nside
    face_e = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix_e = jm_e & (nside - 1)
    iy_e = nside - (jp_e & (nside - 1)) - 1

    # polar caps
    ntt = np.minimum(tt.astype(np.int64), 3)
    tp = tt - ntt
    tmp = nside * np.sqrt(3.0 * (1.0 - za))
    jp_p = np.minimum((tp * tmp).astype(np.int64), nside - 1)
    jm_p = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z >= 0
    face_p = np.where(north, ntt, ntt + 8)
    ix_p = np.where(north, nside - jm_p - 1, jp_p)
    iy_p = np.where(north, nside - jp_p - 1, jm_p)

    equatorial = za <= 2.0 / 3.0
    face = np.where(equatorial, face_e, face_p)
    ix = np.where(equatorial, ix_e, ix_p)
    iy = np.where(equatorial, iy_e, iy_p)

    return face * nside * nside + _spread_bits(ix) + (_spread_bits(iy) << 1)


def _load_map(path):
    """(nside, wco, nobs) for the CO map, or None (one warning) if it can't be read."""
    if path not in _map_cache:
        try:
            with fits.open(path) as hdul:
                hdr = hdul[1].header
                if str(hdr["ORDERING"]).strip().upper() != "NESTED":
                    raise ValueError("expected NESTED ordering")
                data = hdul[1].data
                wco = np.array(data["TEMPERATURE"], dtype=np.float32)
                nobs = np.array(data["N_OBS"], dtype=np.float32)
                _map_cache[path] = (int(hdr["NSIDE"]), wco, nobs)
        except Exception as e:
            print(f"[WARN] can't use CO map {path} ({e}); nh2 will be NaN")
            _map_cache[path] = None
    return _map_cache[path]


def get_wco(ra_deg, dec_deg, map_path, clip_negative=True):
    """W_CO [K km/s] at an ICRS position. NaN outside the survey or for non-finite input."""
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return np.nan

    cmap = _load_map(map_path)
    if cmap is None:
        return np.nan
    nside, wco, nobs = cmap

    gal = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs").galactic
    pix = int(ang2pix_nest(nside, gal.l.deg, gal.b.deg)[0])

    if nobs[pix] <= 0:
        return np.nan
    val = float(wco[pix])
    return max(val, 0.0) if clip_negative else val


def get_nh2(ra_deg, dec_deg, map_path, xco=XCO, clip_negative=True):
    """Return (N_H2 [cm^-2], W_CO [K km/s]); both NaN if there is no CO coverage."""
    w = get_wco(ra_deg, dec_deg, map_path, clip_negative=clip_negative)
    return xco * w, w


# ======================================================================
# Combined entry point
# ======================================================================

def get_nh_estimates(
    ra_deg,
    dec_deg,
    pfile_dir,
    wco_map,
    xco=XCO,
    clip_negative=True,
):
    """
    nh1 and nh2 at (ra_deg, dec_deg), ICRS degrees. Returns a dict:

        nh1          atomic H column, closest HI4PI pixel          [cm^-2]
        nh2          molecular H2 column, X_CO * W_CO              [cm^-2]
        nh1_dist_deg distance from the position to that HI4PI pixel [deg]
        wco          W_CO used for nh2                             [K km/s]

    Any value that cannot be determined is NaN.
    """
    nh1, nh1_dist = get_nh_closest(ra_deg, dec_deg, pfile_dir)
    nh2, wco = get_nh2(ra_deg, dec_deg, wco_map, xco=xco, clip_negative=clip_negative)
    return {"nh1": nh1, "nh2": nh2, "nh1_dist_deg": nh1_dist, "wco": wco}


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(
        description="Print nh1 and nh2 at a position (ICRS degrees)."
    )
    ap.add_argument("ra", type=float, help="RA in degrees")
    ap.add_argument("dec", type=float, help="Dec in degrees")
    ap.add_argument("--wco", required=True, help="path to lambda_wco_dht2001.fits")
    args = ap.parse_args()

    est = get_nh_estimates(
        args.ra, args.dec, tempfile.mkdtemp(prefix="nh_pf_"), args.wco
    )
    print(f"RA={args.ra:.6f} DEC={args.dec:.6f}")
    print(f"  nh1 = {est['nh1']:.3E} cm^-2  (closest HI4PI pixel {est['nh1_dist_deg']:.4f} deg away)")
    print(f"  nh2 = {est['nh2']:.3E} cm^-2  (W_CO = {est['wco']:.2f} K km/s)")
