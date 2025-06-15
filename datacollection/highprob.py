import pandas as pd
import requests
from tqdm import tqdm
import os
import math
import re

# PARAMETERS
CSV_PATH = 'data/GalaxyZoo1/GalaxyZoo1_DR_table2.csv'    # Update with your actual CSV filename
OUT_DIR = 'galaxy_images'
THRESHOLD = 0.85      # Classification probability threshold
MAX_IMAGES = 90000    # Change this as needed
IMAGE_SIZE = 256 # Wider field of view so the galaxy is captured even if off-center
SCALE = .396    # This is the SDSS native pixel scale in arcsec/pixel

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Filter for high-confidence ellipticals and spirals
ellipticals = df[df['P_EL_DEBIASED'] > THRESHOLD]
spirals = df[df['P_CS_DEBIASED'] > THRESHOLD]

# Combine and shuffle (optional)
df_filtered = pd.concat([ellipticals, spirals]).sample(frac=1, random_state=42)
df_filtered = df_filtered.head(MAX_IMAGES)

print(f"Number of galaxies to download: {len(df_filtered)}")

def download_image(ra, dec, label, objid):
    url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={SCALE}&width={IMAGE_SIZE}&height={IMAGE_SIZE}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            fname = f"{label}_{objid}_{float(ra):.4f}_{float(dec):.4f}.jpg"
            fpath = os.path.join(OUT_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download for RA: {ra}, DEC: {dec}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    

def sexagesimal_to_decimal(coord: str, is_ra: bool = None) -> float:
    """
    Convert a sexagesimal string to decimal degrees.

    Parameters
    ----------
    coord : str
        Either  HH:MM:SS.ss   (right ascension, hours)  **or**
        ±DD:MM:SS.ss          (declination, degrees)
    is_ra : bool or None
        * True  – force RA interpretation
        * False – force Dec interpretation
        * None  – auto‑detect: 0 ≤ field0 < 24 ⇒ RA, else Dec

    Returns
    -------
    float
        Angle in decimal degrees.
    """
    # accept :, space, h m s as separators
    parts = [p for p in re.split(r'[ :hms]+', coord.strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"Expected three fields, got {len(parts)} in '{coord}'")

    h_or_deg, m, s = map(float, parts)

    # auto‑detect RA vs Dec if caller didn't say
    if is_ra is None:
        is_ra = 0 <= h_or_deg < 24 and not coord.lstrip().startswith(('+', '-'))

    # sanity checks
    if not (0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"Minutes/seconds out of range in '{coord}'")
    if is_ra:
        if not (0 <= h_or_deg < 24):
            raise ValueError(f"RA hours out of range in '{coord}'")
        dec_deg = (h_or_deg + m/60 + s/3600) * 15        # hours → degrees
    else:
        if not (-90 <= h_or_deg <= 90):
            raise ValueError(f"Dec degrees out of range in '{coord}'")
        dec_deg = math.copysign(abs(h_or_deg) + m/60 + s/3600, h_or_deg)

    return dec_deg

# Loop and download images
success = 0
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):

    try:
        ra = sexagesimal_to_decimal(str(row['RA']))
        dec = sexagesimal_to_decimal(str(row['DEC']))
    except Exception as e:
        print(f"Error converting RA/DEC: {e}")
        continue
        
    # Choose label for filename
    if row['P_EL_DEBIASED'] > THRESHOLD:
        label = "elliptical"
    elif row['P_CS_DEBIASED'] > THRESHOLD:
        label = "spiral"
    else:
        continue

    # Download image
    ok = download_image(ra, dec, label, row['OBJID'])
    if ok:
        success += 1

print(f"Downloaded {success} images to '{OUT_DIR}'")