#!/usr/bin/env python3
"""Script to delete unnecessary files and directories"""
import os
import shutil
from pathlib import Path

def delete_unnecessary_items(root_dir='.', dry_run=True):
    """Delete all unnecessary files and directories"""
    deleted = {
        'cache_dirs': [],
        'build_artifacts': [],
        'ide_files': [],
        'temp_files': []
    }
    
    root = Path(root_dir)
    
    # Patterns to search for
    cache_patterns = ['__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints']
    build_patterns = ['dist', 'build']
    ide_patterns = ['.vscode', '.idea', '.vs']
    temp_patterns = ['.DS_Store', '*.pyc', '*.pyo', '*.pyd', '*.log', '*.bak', '*.tmp']
    
    for item in root.rglob('*'):
        if item.is_dir():
            name = item.name
            # Cache directories
            if name in cache_patterns:
                if not dry_run:
                    try:
                        shutil.rmtree(item)
                        deleted['cache_dirs'].append(str(item))
                    except Exception as e:
                        print(f"Error deleting {item}: {e}")
                else:
                    deleted['cache_dirs'].append(str(item))
            # Build artifacts
            elif name in build_patterns or name.endswith('.egg-info'):
                if not dry_run:
                    try:
                        shutil.rmtree(item)
                        deleted['build_artifacts'].append(str(item))
                    except Exception as e:
                        print(f"Error deleting {item}: {e}")
                else:
                    deleted['build_artifacts'].append(str(item))
            # IDE files
            elif name in ide_patterns:
                if not dry_run:
                    try:
                        shutil.rmtree(item)
                        deleted['ide_files'].append(str(item))
                    except Exception as e:
                        print(f"Error deleting {item}: {e}")
                else:
                    deleted['ide_files'].append(str(item))
        elif item.is_file():
            name = item.name
            # Temp files
            if any(name.endswith(ext) for ext in ['.pyc', '.pyo', '.pyd', '.log', '.bak', '.tmp']) or name == '.DS_Store':
                if not dry_run:
                    try:
                        item.unlink()
                        deleted['temp_files'].append(str(item))
                    except Exception as e:
                        print(f"Error deleting {item}: {e}")
                else:
                    deleted['temp_files'].append(str(item))
    
    return deleted

if __name__ == '__main__':
    import sys
    
    # Check for --execute flag
    dry_run = '--execute' not in sys.argv
    
    if dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files will be deleted")
        print("Add --execute flag to actually delete files")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print("DELETING UNNECESSARY FILES AND DIRECTORIES")
        print("=" * 80)
        print()
    
    deleted = delete_unnecessary_items(dry_run=dry_run)
    
    total_count = 0
    
    if deleted['cache_dirs']:
        print(f"[CACHE DIRECTORIES] ({len(deleted['cache_dirs'])} {'found' if dry_run else 'deleted'}):")
        print("-" * 80)
        for item in sorted(deleted['cache_dirs']):
            print(f"  {'[WOULD DELETE]' if dry_run else '[DELETED]'} {item}")
        print()
        total_count += len(deleted['cache_dirs'])
    
    if deleted['build_artifacts']:
        print(f"[BUILD ARTIFACTS] ({len(deleted['build_artifacts'])} {'found' if dry_run else 'deleted'}):")
        print("-" * 80)
        for item in sorted(deleted['build_artifacts']):
            print(f"  {'[WOULD DELETE]' if dry_run else '[DELETED]'} {item}")
        print()
        total_count += len(deleted['build_artifacts'])
    
    if deleted['ide_files']:
        print(f"[IDE FILES] ({len(deleted['ide_files'])} {'found' if dry_run else 'deleted'}):")
        print("-" * 80)
        for item in sorted(deleted['ide_files']):
            print(f"  {'[WOULD DELETE]' if dry_run else '[DELETED]'} {item}")
        print()
        total_count += len(deleted['ide_files'])
    
    if deleted['temp_files']:
        print(f"[TEMPORARY FILES] ({len(deleted['temp_files'])} {'found' if dry_run else 'deleted'}):")
        print("-" * 80)
        for item in sorted(deleted['temp_files']):
            print(f"  {'[WOULD DELETE]' if dry_run else '[DELETED]'} {item}")
        print()
        total_count += len(deleted['temp_files'])
    
    print("=" * 80)
    if dry_run:
        print(f"TOTAL: {total_count} items would be deleted")
        print("Run with --execute flag to actually delete them")
    else:
        print(f"TOTAL: {total_count} items deleted successfully")
    print("=" * 80)

