import os
import glob
import re
import imageio
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image

def natural_sort_key(s):
    """Sort strings with embedded numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def resize_image(image_path, target_size):
    """Resize image to match the target size."""
    img = Image.open(image_path)
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

def create_video_from_pngs(png_folders, output_file, fps=0.5):
    """Create a video from PNG files in the given folders in sequential order."""
    png_files = []
    
    for folder in png_folders:
        if os.path.isdir(folder):
            folder_images = glob.glob(os.path.join(folder, "*.png"))
            folder_images.sort(key=natural_sort_key)  # Sort each folder's images
            png_files.extend(folder_images)  # Add them sequentially

    if not png_files:
        print(f"No PNG files found in {png_folders}")
        return False

    # Read the first image to get the target size
    first_img = Image.open(png_files[0])
    target_size = first_img.size  # (width, height)

    # Resize all images to match the first image
    images = [resize_image(png, target_size) for png in png_files]

    # Create video clip
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_file, codec="libx264", fps=fps)
    
    print(f"Video created: {output_file}")
    return True

def main():
    # Base directories
    results_dir = "results"
    videos_dir = "videos"

    # Create videos directory if it doesn't exist
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        print(f"Created directory: {videos_dir}")

    # Process each subdirectory in results
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        
        # Process characters first, then chapters
        png_folders = [
            os.path.join(subdir_path, "characters"),
            os.path.join(subdir_path, "chapters")
        ]

        output_file = os.path.join(videos_dir, f"{subdir}.mp4")
        create_video_from_pngs(png_folders, output_file)

if __name__ == "__main__":
    main()

