import argparse
from pathlib import Path
import shutil
from typing import List, Tuple, Optional

def remove_last_lines_from_file(
    file_path: Path, 
    num_lines: int,
    create_backup: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Remove the last N lines from a file.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to modify
    num_lines : int
        Number of lines to remove from the end of the file
    create_backup : bool, optional
        Whether to create a backup of the original file
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        Success status and error message if any
    """
    try:
        # Create backup if requested
        if create_backup:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
            shutil.copy2(file_path, backup_path)
        
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # If file has fewer lines than we want to remove, keep at least one line
        if len(lines) <= num_lines:
            keep_lines = 1
        else:
            keep_lines = len(lines) - num_lines
        
        # Write back all lines except the last num_lines
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines[:keep_lines])
            
        return True, None
    except Exception as e:
        return False, str(e)

def process_directory(
    directory_path: Path, 
    num_lines: int,
    create_backup: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Process all markdown files in a directory and remove the specified
    number of lines from the end of each file.
    
    Parameters
    ----------
    directory_path : Path
        Path to the directory containing markdown files
    num_lines : int
        Number of lines to remove from the end of each file
    create_backup : bool, optional
        Whether to create backups of the original files
        
    Returns
    -------
    Tuple[int, int, List[str]]
        Number of files processed, number of files with errors, 
        and a list of error messages
    """
    processed_count = 0
    error_count = 0
    errors = []
    
    for markdown_file in directory_path.glob('**/*.md'):
        processed_count += 1
        success, error_msg = remove_last_lines_from_file(
            markdown_file, num_lines, create_backup
        )
        
        if not success:
            error_count += 1
            errors.append(f"Error processing {markdown_file}: {error_msg}")
    
    return processed_count, error_count, errors

def main() -> None:
    processed, errors, error_messages = process_directory(
        Path("pg_books"), 9
    )
    
    print(f"\nSummary:")
    print(f"Files processed: {processed}")
    print(f"Files with errors: {errors}")
    
    if errors > 0:
        print("\nErrors encountered:")
        for msg in error_messages:
            print(f" - {msg}")

if __name__ == "__main__":
    main()
