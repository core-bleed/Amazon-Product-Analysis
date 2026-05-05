#!/usr/bin/env python3
"""
Script to download Amazon Question/Answer dataset from Julian McAuley, UCSD
"""

import urllib.request
import urllib.error
from pathlib import Path

# Base URL for the dataset
BASE_URL = "http://jmcauley.ucsd.edu/data/amazon/qa/"

# List of all categories (per-category files)
CATEGORIES = [
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby",
    "Beauty",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
]

# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_file(url, filepath):
    """Download a file from URL to filepath with progress indication."""
    try:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, filepath)
        file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✓ Downloaded: {filepath.name} ({file_size:.2f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"✗ Error downloading {url}: HTTP {e.code}")
        return False
    except urllib.error.URLError as e:
        print(f"✗ Error downloading {url}: {e.reason}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {url}: {e}")
        return False


def main():
    """Download all Amazon Q/A category files."""
    print(f"Downloading Amazon Q/A dataset to: {DATA_DIR}")
    print(f"Total categories: {len(CATEGORIES)}\n")

    downloaded = 0
    failed = 0

    for category in CATEGORIES:
        filename = f"qa_{category}.json.gz"
        url = BASE_URL + filename
        filepath = DATA_DIR / filename

        # Skip if file already exists
        if filepath.exists():
            print(f"⊘ Skipping {filename} (already exists)")
            continue

        if download_file(url, filepath):
            downloaded += 1
        else:
            failed += 1
        print()  # Empty line for readability

    print("=" * 60)
    print("Download complete!")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Files saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
