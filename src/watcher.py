"""
Watches ~/Downloads for new .txt, .pdf, and .docx files and ingests them
into the local ChromaDB knowledge base automatically.

Run with: make watch
"""

import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import ingest, SUPPORTED_EXTENSIONS

WATCH_DIR = Path.home() / "Downloads"


def is_fully_written(path: Path, wait: float = 1.5) -> bool:
    """Return True if the file size has stabilised — guards against partial downloads."""
    try:
        size_before = path.stat().st_size
        time.sleep(wait)
        return path.stat().st_size == size_before
    except FileNotFoundError:
        return False


class DownloadsHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return
        print(f"\n[watcher] New file detected: {path.name}")
        if not is_fully_written(path):
            print(f"[watcher] File still being written, skipping: {path.name}")
            return
        ingest(str(path))


if __name__ == "__main__":
    print(f"Watching {WATCH_DIR} for new .txt, .pdf, .docx files...")
    print("Press Ctrl+C to stop.\n")

    observer = Observer()
    observer.schedule(DownloadsHandler(), str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
