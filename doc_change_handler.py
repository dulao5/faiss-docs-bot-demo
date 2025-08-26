from watchdog.events import FileSystemEventHandler
from threading import Timer, Lock
import time

class DocChangeHandler(FileSystemEventHandler):
    """
    A file system event handler that manages document change events with cooldown period.
    
    Features:
    - Debounces rapid successive changes
    - Ignores hidden files and non-markdown files
    - Thread-safe operation
    - Prevents concurrent rebuilds
    """
    
    def __init__(self, rebuild_callback, cooldown=5.0, watch_extensions=('.md',)):
        """
        Initialize the handler.
        
        Args:
            rebuild_callback: Function to call when rebuild is needed
            cooldown: Number of seconds to wait before allowing another rebuild
        """
        self.rebuild_callback = rebuild_callback
        self.cooldown = cooldown
        self.timer = None
        self.lock = Lock()
        self.last_rebuild_time = 0
        self.is_rebuilding = False
        self.extensions = watch_extensions

    def on_any_event(self, event):
        # Ignore hidden files and non-markdown files
        if (event.is_directory or 
            not event.src_path.endswith(self.extensions) or
            '/.' in event.src_path or
            event.src_path.startswith('.')):
            return
            
        current_time = time.time()
        with self.lock:
            # Skip if currently rebuilding or within cooldown period
            if self.is_rebuilding or (current_time - self.last_rebuild_time) < self.cooldown:
                return

            if self.timer:
                self.timer.cancel()
            
            def protected_rebuild():
                try:
                    self.is_rebuilding = True
                    self.rebuild_callback()
                finally:
                    self.is_rebuilding = False
                    self.last_rebuild_time = time.time()

            self.timer = Timer(self.cooldown, protected_rebuild)
            self.timer.start()