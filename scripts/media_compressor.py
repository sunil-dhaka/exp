#!/usr/bin/env python3
"""
Media File Compression Utility

This script compresses media files in a directory while preserving
the original dimensions and folder structure, with Unix-like naming conventions.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Callable
from abc import ABC, abstractmethod
import argparse
import concurrent.futures
import threading
import re
import subprocess
import unicodedata
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add a lock for thread-safe logging
log_lock = threading.Lock()

class FFmpegNotFoundError(Exception):
    """Exception raised when FFmpeg tools are not found."""
    pass

def get_file_size_mb(file_path: Path) -> float:
    """
    Get the size of a file in megabytes.
    
    Parameters
    ----------
    file_path : Path
        Path to the file
        
    Returns
    -------
    float
        File size in megabytes
    """
    return os.path.getsize(file_path) / (1024 * 1024)

def normalize_name(name: str) -> str:
    """
    Convert a filename or folder name to Unix-like snake_case.
    
    Parameters
    ----------
    name : str
        Original name
        
    Returns
    -------
    str
        Normalized name in snake_case format
    """
    # Remove extension if present
    base_name = name
    extension = ""
    
    if "." in name and not name.startswith("."):
        base_name, extension = name.rsplit(".", 1)
        extension = f".{extension.lower()}"
    
    # Replace spaces and special chars with underscores
    result = re.sub(r'[^a-zA-Z0-9]', '_', base_name)
    # Replace consecutive underscores with single underscore
    result = re.sub(r'_+', '_', result)
    # Remove leading/trailing underscores
    result = result.strip('_')
    # Convert to lowercase
    result = result.lower()
    
    # If the result is empty (e.g., all special chars), use 'file'
    if not result:
        result = "file"
    
    # Add extension back if it was present
    return f"{result}{extension}"

def normalize_path(path: Path) -> Path:
    """
    Normalize all parts of a path to Unix-like snake_case.
    
    Parameters
    ----------
    path : Path
        Original path
        
    Returns
    -------
    Path
        Path with all parts normalized to snake_case
    """
    # Split the path into parts
    parts = path.parts
    
    # Normalize each part except the drive (if present)
    if path.drive:
        normalized_parts = [path.drive] + [normalize_name(part) for part in parts[1:]]
    else:
        normalized_parts = [normalize_name(part) for part in parts]
    
    # Reconstruct the path
    return Path(*normalized_parts)

class MediaCompressor(ABC):
    """
    Abstract base class for media file compressors.
    
    This class defines the interface for different media compressors.
    """
    
    @abstractmethod
    def compress(self, input_path: Path, output_path: Path) -> bool:
        """
        Compress a media file.
        
        Parameters
        ----------
        input_path : Path
            Path to the input file
        output_path : Path
            Path where the compressed file will be saved
            
        Returns
        -------
        bool
            True if compression was successful, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> Set[str]:
        """
        Returns the file formats supported by this compressor.
        
        Returns
        -------
        Set[str]
            Set of file extensions this compressor can handle
        """
        pass
    
    def _check_compression_ratio(self, input_path: Path, output_path: Path, target_ratio: float = 0.1) -> bool:
        """
        Check if the achieved compression ratio meets the target.
        
        Parameters
        ----------
        input_path : Path
            Original file path
        output_path : Path
            Compressed file path
        target_ratio : float, optional
            Target compression ratio (output size / input size), by default 0.1
            
        Returns
        -------
        bool
            True if compression is satisfactory, False if more compression is needed
        """
        if not output_path.exists():
            return False
            
        input_size = get_file_size_mb(input_path)
        output_size = get_file_size_mb(output_path)
        
        if input_size == 0:
            return True
            
        actual_ratio = output_size / input_size
        
        with log_lock:
            logger.info(f"File: {input_path.name}, Original: {input_size:.2f}MB, " 
                       f"Compressed: {output_size:.2f}MB, Ratio: {actual_ratio:.2%}")
        
        # Return True if we've met or exceeded target compression (smaller ratio is better)
        return actual_ratio <= target_ratio


class ImageCompressor(MediaCompressor):
    """
    Compressor for image files.
    """
    
    def __init__(self, quality: int = 30, target_ratio: float = 0.1):
        """
        Initialize the image compressor.
        
        Parameters
        ----------
        quality : int, optional
            Compression quality (0-100), by default 30
        target_ratio : float, optional
            Target compression ratio (output/input), by default 0.1
        """
        self._quality = quality
        self._target_ratio = target_ratio
        self._supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    @property
    def supported_formats(self) -> Set[str]:
        """
        Returns the file formats supported by this compressor.
        
        Returns
        -------
        Set[str]
            Set of file extensions this compressor can handle
        """
        return self._supported_formats
    
    def compress(self, input_path: Path, output_path: Path) -> bool:
        """
        Compress an image file.
        
        Parameters
        ----------
        input_path : Path
            Path to the input image
        output_path : Path
            Path where the compressed image will be saved
            
        Returns
        -------
        bool
            True if compression was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Import here to avoid dependency issues if only video compression is needed
            from PIL import Image
            import io
            
            # Start with the initial quality setting
            current_quality = self._quality
            max_attempts = 3
            attempts = 0
            
            with Image.open(input_path) as img:
                # For transparent PNGs, convert to JPEG with white background if significantly smaller
                if input_path.suffix.lower() == '.png' and 'A' in img.getbands():
                    # Try compressing as PNG first
                    png_output = output_path
                    self._save_optimized_image(img, png_output, format_name='PNG')
                    
                    # If PNG is still too large, try JPEG with white background
                    if not self._check_compression_ratio(input_path, png_output, self._target_ratio):
                        # Convert to JPEG with white background
                        jpg_output = output_path.with_suffix('.jpg')
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        self._save_optimized_image(background, jpg_output, format_name='JPEG', quality=current_quality)
                        
                        # If JPEG is smaller than PNG, use it
                        if jpg_output.exists() and get_file_size_mb(jpg_output) < get_file_size_mb(png_output):
                            if os.path.exists(png_output):
                                os.remove(png_output)
                            shutil.move(jpg_output, output_path)
                else:
                    # For other formats, try progressive quality reduction
                    while attempts < max_attempts:
                        # Preserve the original format
                        format_name = input_path.suffix.lstrip('.').upper()
                        if format_name == 'JPG':
                            format_name = 'JPEG'
                            
                        # Try to resize the image if it's very large and we've already tried reducing quality
                        if attempts > 0:
                            # Scale down by sqrt(target_ratio) in each dimension to achieve approximate target ratio
                            # while maintaining aspect ratio
                            scale_factor = 0.7 if attempts == 1 else 0.5
                            new_width = int(img.width * scale_factor)
                            new_height = int(img.height * scale_factor)
                            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                            self._save_optimized_image(resized_img, output_path, format_name, quality=current_quality)
                        else:
                            self._save_optimized_image(img, output_path, format_name, quality=current_quality)
                        
                        # Check if we've achieved the target compression ratio
                        if self._check_compression_ratio(input_path, output_path, self._target_ratio):
                            break
                            
                        # Try with a more aggressive quality setting
                        current_quality = max(10, current_quality - 20)
                        attempts += 1
            
            with log_lock:
                logger.info(f"Compressed image: {input_path}")
            return True
            
        except ImportError:
            with log_lock:
                logger.error("PIL/Pillow not installed. Cannot compress images.")
                logger.error("Install with: pip install Pillow")
            return False
        except Exception as e:
            with log_lock:
                logger.error(f"Error compressing image {input_path}: {e}")
            return False
            
    def _save_optimized_image(self, img, output_path: Path, format_name: str, quality: int = None) -> None:
        """
        Save an image with optimized settings depending on the format.
        
        Parameters
        ----------
        img : PIL.Image
            The image to save
        output_path : Path
            Path where to save the image
        format_name : str
            Format name (JPEG, PNG, etc.)
        quality : int, optional
            Quality setting for lossy formats
        """
        if format_name == 'JPEG':
            img.save(output_path, format=format_name, quality=quality, optimize=True, progressive=True)
        elif format_name == 'PNG':
            img.save(output_path, format=format_name, optimize=True, compress_level=9)
        else:
            img.save(output_path, format=format_name, optimize=True)


class VideoCompressor(MediaCompressor):
    """
    Compressor for video files.
    """
    
    def __init__(self, crf: int = 28, target_ratio: float = 0.1):
        """
        Initialize the video compressor.
        
        Parameters
        ----------
        crf : int, optional
            Constant Rate Factor (0-51, higher means more compression), by default 28
        target_ratio : float, optional
            Target compression ratio (output/input), by default 0.1
        """
        self._crf = crf
        self._target_ratio = target_ratio
        self._supported_formats = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        self._check_ffmpeg()
    
    @property
    def supported_formats(self) -> Set[str]:
        """
        Returns the file formats supported by this compressor.
        
        Returns
        -------
        Set[str]
            Set of file extensions this compressor can handle
        """
        return self._supported_formats
    
    def _check_ffmpeg(self) -> None:
        """
        Check if FFmpeg and ffprobe are available in the system.
        
        Raises
        ------
        FFmpegNotFoundError
            If FFmpeg tools are not found
        """
        try:
            # Use subprocess.run with capture_output to avoid displaying the version info
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise FFmpegNotFoundError(
                "FFmpeg/ffprobe not found. Please install FFmpeg and make sure it's in your PATH."
            )
        logger.info("FFmpeg tools found and ready")
    
    def _is_valid_video(self, file_path: Path) -> bool:
        """
        Check if a file is a valid video file.
        
        Parameters
        ----------
        file_path : Path
            Path to the file to check
            
        Returns
        -------
        bool
            True if the file is a valid video, False otherwise
        """
        try:
            # Use a simple ffprobe command to check if the file is a valid video
            # -v error: only show errors
            # -show_entries format=duration: only check the duration
            # -of default=noprint_wrappers=1: don't print the section wrappers
            result = subprocess.run(
                [
                    'ffprobe', 
                    '-v', 'error',
                    '-select_streams', 'v:0',  # Select the first video stream
                    '-show_entries', 'stream=codec_type',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                timeout=10  # Add a timeout to prevent hanging
            )
            
            # Check if the output contains "video"
            return result.returncode == 0 and "video" in result.stdout
            
        except Exception:
            return False
    
    def _check_file_type(self, file_path: Path) -> str:
        """
        Try to determine the actual file type (MIME type).
        
        Parameters
        ----------
        file_path : Path
            Path to the file
            
        Returns
        -------
        str
            MIME type of the file or 'unknown'
        """
        try:
            # Try using the file command (common on Unix-like systems)
            result = subprocess.run(
                ['file', '--mime-type', '-b', str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            return 'unknown'
        except Exception:
            return 'unknown'
    
    def _get_video_info(self, input_path: Path) -> Optional[Dict[str, any]]:
        """
        Get video information using ffprobe directly via subprocess.
        
        Parameters
        ----------
        input_path : Path
            Path to the video file
            
        Returns
        -------
        Optional[Dict[str, any]]
            Dictionary containing video information or None if ffprobe failed
        """
        import json
        
        # First check if it's a valid video file
        if not self._is_valid_video(input_path):
            return None
            
        # Create a temporary file with a simple name to avoid issues with special characters
        with tempfile.NamedTemporaryFile(suffix=input_path.suffix) as temp_file:
            try:
                # Copy the file to the temporary file
                shutil.copy2(input_path, temp_file.name)
                
                # Use ffprobe to get video information
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    temp_file.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return None
                    
                return json.loads(result.stdout)
                
            except Exception:
                return None
    
    def compress(self, input_path: Path, output_path: Path) -> bool:
        """
        Compress a video file.
        
        Parameters
        ----------
        input_path : Path
            Path to the input video
        output_path : Path
            Path where the compressed video will be saved
            
        Returns
        -------
        bool
            True if compression was successful, False otherwise
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First verify if this is actually a valid video file
        if not self._is_valid_video(input_path):
            # Get the actual file type for logging
            file_type = self._check_file_type(input_path)
            
            with log_lock:
                logger.warning(f"File {input_path} has video extension but appears to be {file_type}. Copying instead.")
            
            # Just copy the file
            try:
                shutil.copy2(input_path, output_path)
                return True
            except Exception as e:
                with log_lock:
                    logger.error(f"Error copying file {input_path}: {str(e)}")
                return False
        
        try:
            # Get video info
            video_info = self._get_video_info(input_path)
            
            if not video_info:
                with log_lock:
                    logger.warning(f"Could not get video info for {input_path}. Using default compression settings.")
            
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(suffix=output_path.suffix, delete=False) as temp_output_file:
                temp_output_path = Path(temp_output_file.name)
                
                # Start with initial settings
                current_crf = self._crf
                
                # Prepare ffmpeg command
                cmd = ['ffmpeg', '-i', str(input_path)]
                
                # Add video codec and compression parameters
                cmd.extend([
                    '-c:v', 'libx264',
                    '-crf', str(current_crf),
                    '-preset', 'medium',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    '-y',
                    str(temp_output_path)
                ])
                
                # Execute ffmpeg
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    with log_lock:
                        logger.error(f"FFmpeg error: {process.stderr}")
                    
                    # Fallback to just copying the file
                    with log_lock:
                        logger.warning(f"Compression failed for {input_path}. Copying instead.")
                    
                    shutil.copy2(input_path, output_path)
                    return True
                
                # Check if the output is valid and smaller
                if temp_output_path.exists() and self._is_valid_video(temp_output_path):
                    # Check compression ratio
                    if self._check_compression_ratio(input_path, temp_output_path, self._target_ratio):
                        shutil.copy2(temp_output_path, output_path)
                        with log_lock:
                            logger.info(f"Successfully compressed video: {input_path}")
                        return True
                    else:
                        # If we couldn't achieve target compression, just copy the original
                        with log_lock:
                            logger.warning(f"Could not achieve target compression for {input_path}. Copying original.")
                        shutil.copy2(input_path, output_path)
                        return True
                else:
                    # Output is invalid or doesn't exist, copy original
                    with log_lock:
                        logger.warning(f"Output file invalid for {input_path}. Copying original.")
                    shutil.copy2(input_path, output_path)
                    return True
                    
        except Exception as e:
            with log_lock:
                logger.error(f"Error processing video {input_path}: {str(e)}")
            
            # Fallback to copying the file
            try:
                shutil.copy2(input_path, output_path)
                with log_lock:
                    logger.warning(f"Compression failed, copied original file: {input_path}")
                return True
            except Exception as copy_err:
                with log_lock:
                    logger.error(f"Error copying file {input_path}: {str(copy_err)}")
                return False
        finally:
            # Clean up any temporary files
            if 'temp_output_path' in locals() and temp_output_path.exists():
                try:
                    os.remove(temp_output_path)
                except Exception:
                    pass


class CompressionService:
    """
    Service that coordinates the compression of media files.
    """
    
    def __init__(self, max_workers: int = 10, target_ratio: float = 0.1, normalize_names: bool = True):
        """
        Initialize the compression service with specific compressors.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker threads, by default 10
        target_ratio : float, optional
            Target compression ratio (output/input), by default 0.1
        normalize_names : bool, optional
            Whether to normalize filenames and folder names, by default True
        """
        self.max_workers = max_workers
        self.target_ratio = target_ratio
        self.normalize_names = normalize_names
        self.compressors: List[MediaCompressor] = [
            ImageCompressor(target_ratio=target_ratio),
            VideoCompressor(target_ratio=target_ratio)
        ]
        
        # Map file extensions to their appropriate compressor
        self.extension_map: Dict[str, MediaCompressor] = {}
        for compressor in self.compressors:
            for ext in compressor.supported_formats:
                self.extension_map[ext] = compressor
    
    def process_file(self, input_file: Path, output_file: Path) -> bool:
        """
        Process a single file.
        
        Parameters
        ----------
        input_file : Path
            Path to the input file
        output_file : Path
            Path where the compressed file will be saved
            
        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        # Get the file extension in lowercase
        file_ext = input_file.suffix.lower()
        
        # Check if we have a compressor for this file type
        if file_ext in self.extension_map:
            compressor = self.extension_map[file_ext]
            return compressor.compress(input_file, output_file)
        else:
            # For unsupported files, just copy them
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_file, output_file)
                with log_lock:
                    logger.info(f"Copied non-media file: {input_file} -> {output_file}")
                return True
            except Exception as e:
                with log_lock:
                    logger.error(f"Error copying {input_file}: {e}")
                return False
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Tuple[int, int]:
        """
        Process all media files in the input directory and its subdirectories.
        
        Parameters
        ----------
        input_dir : Path
            Path to the input directory
        output_dir : Path
            Path where compressed files will be saved
            
        Returns
        -------
        Tuple[int, int]
            Count of (successful compressions, failures)
        """
        success_count = 0
        failure_count = 0
        
        # Collect all files to process
        files_to_process = []
        
        for root, _, files in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_dir)
            
            # Create normalized output directory path if normalization is enabled
            if self.normalize_names and rel_path != '.':
                norm_path_parts = [normalize_name(part) for part in Path(rel_path).parts]
                current_output_dir = output_dir.joinpath(*norm_path_parts)
            else:
                current_output_dir = output_dir / rel_path
            
            for file_name in files:
                input_file = Path(root) / file_name
                
                # Normalize filename if needed
                if self.normalize_names:
                    output_file_name = normalize_name(file_name)
                else:
                    output_file_name = file_name
                    
                output_file = current_output_dir / output_file_name
                files_to_process.append((input_file, output_file))
        
        total_files = len(files_to_process)
        with log_lock:
            logger.info(f"Found {total_files} files to process")
        
        # Process files in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_file, input_file, output_file): (input_file, output_file) 
                for input_file, output_file in files_to_process
            }
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                input_file, output_file = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    with log_lock:
                        logger.error(f"Exception while processing {input_file}: {e}")
                    failure_count += 1
                
                # Update progress
                completed += 1
                if completed % 10 == 0 or completed == total_files:
                    with log_lock:
                        logger.info(f"Progress: {completed}/{total_files} files processed")
        
        return success_count, failure_count


def main() -> None:
    """
    Main function to parse arguments and run the compression.
    """
    parser = argparse.ArgumentParser(description="Compress media files while preserving folder structure.")
    parser.add_argument("input_dir", help="Path to the directory containing media files")
    parser.add_argument("-o", "--output-dir", help="Output directory (defaults to input_dir + '_compressed')")
    parser.add_argument("-w", "--workers", type=int, default=10, 
                        help="Maximum number of worker threads (default: 10)")
    parser.add_argument("-r", "--ratio", type=float, default=0.1, 
                        help="Target compression ratio (default: 0.1 = 10%% of original size)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable filename and folder normalization to snake_case")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        # Normalize the input directory name for the output folder
        if not args.no_normalize:
            input_dir_name = normalize_name(input_dir.name)
            output_dir = input_dir.parent / f"{input_dir_name}_compressed"
        else:
            output_dir = input_dir.parent / f"{input_dir.name}_compressed"
    
    # Create compression service
    service = CompressionService(
        max_workers=args.workers, 
        target_ratio=args.ratio,
        normalize_names=not args.no_normalize
    )
    
    logger.info(f"Starting compression from {input_dir} to {output_dir}")
    logger.info(f"Using {args.workers} worker threads with target ratio of {args.ratio:.1%}")
    logger.info(f"File and folder name normalization: {'Disabled' if args.no_normalize else 'Enabled'}")
    
    # Process the directory
    success, failure = service.process_directory(input_dir, output_dir)
    
    # Log results
    logger.info(f"Compression complete:")
    logger.info(f"  - Successfully processed: {success} files")
    logger.info(f"  - Failed to process: {failure} files")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
