#!/usr/bin/env python3
"""
Download all parts for clean_0 from AriaGen2PilotDataset
"""

import json
import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# Load the URLs from the JSON file
with open('AriaGen2PilotDataset_download_urls.json', 'r') as f:
    data = json.load(f)

# Get clean_0 data
clean_0_data = data['sequences']['clean_0']

# Create output directory
output_dir = Path('clean_0_data')
output_dir.mkdir(exist_ok=True)

def calculate_sha1(file_path):
    """Calculate SHA1 checksum of a file"""
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha1.update(chunk)
    return sha1.hexdigest()

def download_file(url, filename, expected_sha1):
    """Download a file and verify checksum"""
    file_path = output_dir / filename
    
    # Skip if file already exists and has correct checksum
    if file_path.exists():
        print(f"File exists: {filename}")
        actual_sha1 = calculate_sha1(file_path)
        if actual_sha1 == expected_sha1:
            print(f"  ✓ Checksum verified: {filename}")
            return True
        else:
            print(f"  ✗ Checksum mismatch for {filename}, re-downloading...")
            file_path.unlink()
    
    try:
        print(f"\nDownloading: {filename}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify checksum
        actual_sha1 = calculate_sha1(file_path)
        if actual_sha1 == expected_sha1:
            print(f"  ✓ Checksum verified: {filename}")
            return True
        else:
            print(f"  ✗ Checksum mismatch!")
            print(f"    Expected: {expected_sha1}")
            print(f"    Got:      {actual_sha1}")
            file_path.unlink()
            return False
    
    except Exception as e:
        print(f"  ✗ Error downloading {filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        return False

# Download all parts
print("=" * 70)
print("Downloading AriaGen2PilotDataset clean_0 parts")
print("=" * 70)

failed_downloads = []
total_size = 0

for part_name, part_info in clean_0_data.items():
    filename = part_info['filename']
    url = part_info['download_url']
    sha1sum = part_info['sha1sum']
    file_size = part_info['file_size_bytes']
    
    total_size += file_size
    
    if not download_file(url, filename, sha1sum):
        failed_downloads.append(filename)

print("\n" + "=" * 70)
print("Download Summary")
print("=" * 70)
print(f"Total size: {total_size / (1024**3):.2f} GB")
print(f"Output directory: {output_dir.absolute()}")

if failed_downloads:
    print(f"\n✗ Failed downloads ({len(failed_downloads)}):")
    for filename in failed_downloads:
        print(f"  - {filename}")
else:
    print("\n✓ All files downloaded successfully!")
