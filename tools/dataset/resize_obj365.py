"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm


def resize_image_and_update_annotations(image_path, annotations, output_image_path, max_size=640, verbose=False):
    # if verbose:
    #     print(f"Processing image: {image_path}")
    try:
        # Kiểm tra file có tồn tại
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return None
        
        with Image.open(image_path) as img:
            w, h = img.size
            
            # Kiểm tra xem có cần resize không
            needs_resize = max(w, h) > max_size
            
            if needs_resize:
                scale = max_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                if verbose:
                    print(f"Resizing image to width={new_w}, height={new_h}")

                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                if verbose:
                    print(f"Original size: ({w}, {h}), New size: ({new_w}, {new_h})")

                # Update annotations with scale
                for ann in annotations:
                    ann["area"] = ann["area"] * (scale**2)
                    ann["bbox"] = [coord * scale for coord in ann["bbox"]]
                    if "orig_size" in ann:
                        ann["orig_size"] = (new_w, new_h)
                    if "size" in ann:
                        ann["size"] = (new_w, new_h)
            else:
                # Không cần resize, nhưng vẫn update metadata
                new_w, new_h = w, h
                if verbose:
                    print(f"Image already small enough ({w}, {h}) - no resize needed")
            
            # ✓ LUU ANH RA OUTPUT DUÙ CÓ RESIZE HOẶC KHÔNG
            img.save(output_image_path)
            if verbose:
                print(f"Image saved: {output_image_path}")

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        print(f"   This image and its annotations will be skipped")
        return None

    return annotations, new_w, new_h, needs_resize


def load_existing_data(output_annotation_path):
    """Load existing annotations if file exists"""
    if os.path.isfile(output_annotation_path):
        try:
            with open(output_annotation_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Warning: Could not load existing annotation file: {e}")
    return None


def get_processed_files(output_image_path):
    """Get list of already processed image files"""
    if not os.path.isdir(output_image_path):
        return set()
    return set(os.listdir(output_image_path))


def resize_images_and_update_annotations(image_dir, annotation_path, output_image_path, output_annotation_path, max_size=640, num_workers=4, verbose=False, resume=False):
    print(f"Starting to resize images and update annotations")
    if resume:
        print(f"Resume mode enabled - will skip already processed files")
    
    json_file = annotation_path
    if not os.path.isfile(json_file):
        print(f"Error: JSON file not found at {json_file}")
        return

    # Create output directories if they don't exist
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)

    print(f"Loading JSON file: {json_file}")
    with open(json_file, "r") as f:
        data = json.load(f)
    print("JSON file loaded.")

    # Load existing data if resume mode is enabled
    existing_data = None
    processed_files = set()
    if resume:
        existing_data = load_existing_data(output_annotation_path)
        processed_files = get_processed_files(output_image_path)
        print(f"Found {len(processed_files)} already processed files")

    print("Preparing image annotations mapping...")
    image_annotations = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)
    print("Image annotations mapping prepared.")

    # Filter images to process if resume mode is enabled
    images_to_process = data["images"]
    if resume and processed_files:
        images_to_process = [
            img for img in data["images"] 
            if os.path.basename(img["file_name"]) not in processed_files
        ]
        skipped_count = len(data["images"]) - len(images_to_process)
        print(f"Skipping {skipped_count} already processed images")

    def process_image(image_info):
        basename = os.path.basename(image_info["file_name"])
        image_path = os.path.join(image_dir, image_info["file_name"])
        output_full_path = os.path.join(output_image_path, basename)
        results = resize_image_and_update_annotations(
            image_path, image_annotations[image_info["id"]], output_full_path, max_size, verbose
        )
        if results is None:
            updated_annotations, new_w, new_h, resized = None, None, None, None
        else:
            updated_annotations, new_w, new_h, resized = results
        return image_info, updated_annotations, new_w, new_h, resized

    print(f"Processing {len(images_to_process)} images with {num_workers} worker threads...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_image, images_to_process), total=len(images_to_process), desc="Processing images"))
    print("Image processing completed.")

    # Load existing images and annotations if resume mode
    if resume and existing_data:
        new_images = existing_data.get("images", [])
        new_annotations = existing_data.get("annotations", [])  # ✓ Load existing annotations
        processed_image_ids = {img["id"] for img in new_images}
        print(f"Loaded {len(new_images)} images and {len(new_annotations)} annotations from existing file")
    else:
        new_images = []
        new_annotations = []
        processed_image_ids = set()
    
    failed_images = []
    failed_image_ids = set()

    print("Updating image and annotation data...")
    for image_info, updated_annotations, new_w, new_h, resized in tqdm(results, desc="Updating annotations"):
        if updated_annotations is not None:
            image_info["width"] = new_w
            image_info["height"] = new_h
            image_annotations[image_info["id"]] = updated_annotations
            
            # Update file_name to reflect output directory
            image_info["file_name"] = os.path.basename(image_info["file_name"])

            # Only add if not already processed (in resume mode)
            if image_info["id"] not in processed_image_ids:
                new_images.append(image_info)
                new_annotations.extend(updated_annotations)  # ✓ Chỉ add annotations cho ảnh mới
            else:
                print(f"⚠ Warning: Image {image_info['file_name']} already exists in output, skipping...")
        else:
            # Ảnh bị lỗi - ghi lại để bỏ qua
            failed_images.append(image_info["file_name"])
            failed_image_ids.add(image_info["id"])
    
    # Loại bỏ các annotations thuộc về ảnh bị lỗi từ cả existing và new
    final_image_ids = {img["id"] for img in new_images}
    new_annotations = [ann for ann in new_annotations if ann["image_id"] in final_image_ids]
    
    print(f"\n{'='*60}")
    print(f"Total images in original dataset: {len(data['images'])}")
    print(f"Successfully processed images: {len(new_images)}")
    print(f"Failed/Skipped images: {len(failed_images)}")
    print(f"Total annotations updated: {len(new_annotations)}")
    
    if failed_images:
        print(f"\n⚠ Failed images (will be excluded):")
        for i, img_name in enumerate(failed_images[:10], 1):
            print(f"  {i}. {img_name}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print(f"{'='*60}\n")

    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"],
    }

    print("Saving new training annotations...")
    with open(output_annotation_path, "w") as f:
        json.dump(new_data, f)
    print(f"New JSON file saved to {output_annotation_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resize images and update dataset annotations."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Path to the JSON annotation file",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        required=True,
        help="Output directory for resized images",
    )
    parser.add_argument(
        "--output_annotation_path",
        type=str,
        required=True,
        help="Output path for the JSON annotation file",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=640,
        help="Maximum size for the longer side of the image (default: 640)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker threads for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run - skip already processed files",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    image_path = args.image_path
    annotation_path = args.annotation_path
    output_image_path = args.output_image_path
    output_annotation_path = args.output_annotation_path
    max_size = args.max_size
    num_workers = args.num_workers
    verbose = args.verbose
    resume = args.resume

    print(f"Processing images from: {image_path}")
    print(f"Using annotation file: {annotation_path}")
    print(f"Output images to: {output_image_path}")
    print(f"Output annotation to: {output_annotation_path}")
    if resume:
        print(f"Resume mode: ON")
    resize_images_and_update_annotations(
        image_dir=image_path, 
        annotation_path=annotation_path,
        output_image_path=output_image_path,
        output_annotation_path=output_annotation_path,
        max_size=max_size, 
        num_workers=num_workers,
        verbose=verbose,
        resume=resume
    )
    print("Processing completed.")


if __name__ == "__main__":
    main()
