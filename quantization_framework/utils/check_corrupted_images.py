"""
Utility to find and optionally remove corrupted images from GTSRB dataset.

Usage:
    # Find corrupted images
    python quantization_framework/utils/check_corrupted_images.py --data-path ./data/Train

    # Remove corrupted images
    python quantization_framework/utils/check_corrupted_images.py --data-path ./data/Train --remove
"""

import os
from PIL import Image
from tqdm import tqdm
import argparse


def check_directory(directory):
    """
    Check all images in directory and subdirectories for corruption.

    Args:
        directory: Root directory to scan

    Returns:
        List of (filepath, error_message) tuples for corrupted files
    """
    corrupted = []
    total_checked = 0

    # Collect all image files first
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp')):
                image_files.append(os.path.join(root, file))

    # Check each image with progress bar
    print(f"Found {len(image_files)} image files to check...")

    for filepath in tqdm(image_files, desc="Checking images"):
        total_checked += 1
        try:
            # Try to open and verify the image
            with Image.open(filepath) as img:
                img.verify()  # Verify it's a valid image

            # Also try to actually load it (verify doesn't catch all issues)
            with Image.open(filepath) as img:
                img.load()  # Force loading of image data

        except Exception as e:
            corrupted.append((filepath, str(e)))

    return corrupted, total_checked


def main():
    parser = argparse.ArgumentParser(
        description='Check for corrupted images in dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check GTSRB training data
  python %(prog)s --data-path ./data/Train

  # Check and remove corrupted files
  python %(prog)s --data-path ./data/Train --remove

  # Check test data
  python %(prog)s --data-path ./data/Test
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/Train',
        help='Path to directory to check (default: ./data/Train)'
    )

    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove corrupted files (default: only report them)'
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.data_path):
        print(f"Error: Directory not found: {args.data_path}")
        return 1

    print(f"{'='*70}")
    print(f"Checking for corrupted images in: {args.data_path}")
    print(f"{'='*70}\n")

    # Scan directory
    corrupted, total_checked = check_directory(args.data_path)

    # Report results
    print(f"\n{'='*70}")
    print(f"Scan Complete")
    print(f"{'='*70}")
    print(f"Total images checked: {total_checked}")
    print(f"Corrupted images found: {len(corrupted)}")

    if corrupted:
        print(f"\n{'='*70}")
        print(f"Corrupted Files:")
        print(f"{'='*70}")

        for filepath, error in corrupted:
            # Show relative path if possible
            try:
                rel_path = os.path.relpath(filepath, args.data_path)
                print(f"\n  File: {rel_path}")
            except:
                print(f"\n  File: {filepath}")
            print(f"  Error: {error}")

        # Handle removal
        if args.remove:
            print(f"\n{'='*70}")
            print(f"Removing {len(corrupted)} corrupted files...")
            print(f"{'='*70}")

            removed_count = 0
            failed_count = 0

            for filepath, _ in corrupted:
                try:
                    os.remove(filepath)
                    removed_count += 1
                    print(f"  ✓ Removed: {os.path.relpath(filepath, args.data_path)}")
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ Failed to remove: {filepath}")
                    print(f"    Error: {e}")

            print(f"\nSummary: {removed_count} removed, {failed_count} failed")

            if removed_count > 0:
                print(f"\n{'='*70}")
                print(f"✓ Dataset cleaned! You can now resume training.")
                print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"To remove these files, run with --remove flag:")
            print(f"python {os.path.basename(__file__)} --data-path {args.data_path} --remove")
            print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"✓ No corrupted images found! Dataset is clean.")
        print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    exit(main())
