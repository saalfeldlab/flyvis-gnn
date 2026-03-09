import os
import re
import subprocess


def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect the last fully completed batch from saved artifacts."""
    found_iters = set()

    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    if os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            match = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if match:
                found_iters.add(int(match.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)
    batch_start = ((last_iter - 1) // n_parallel) * n_parallel + 1
    batch_iters = set(range(batch_start, batch_start + n_parallel))

    if batch_iters.issubset(found_iters):
        resume_at = batch_start + n_parallel
    else:
        resume_at = batch_start

    return resume_at


def is_git_repo(path):
    """Check if path is inside a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=path, capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def get_modified_code_files(root_dir, code_files):
    """Return list of code_files that have uncommitted changes."""
    modified = []
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed = set(result.stdout.strip().splitlines())
        result2 = subprocess.run(
            ['git', 'diff', '--name-only', '--cached'],
            cwd=root_dir, capture_output=True, text=True, timeout=10
        )
        changed.update(result2.stdout.strip().splitlines())
        for f in code_files:
            if f in changed:
                modified.append(f)
    except Exception:
        pass
    return modified
