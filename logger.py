import yaml
import sys
import datetime

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

original_stdout = sys.stdout

class Logger(object):
    def __init__(self, file):
        self.file = file
        # Buffer for handling multi-line messages across write calls (e.g., print adds newline automatically)
        self.buffer = ""

    def write(self, message):
        # Buffer the message, handle newlines across calls
        self.buffer += message
        
        # Split buffer by newline, handle complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)  # Split only at the first newline
            if line.strip() == "":  # Skip empty lines
                self._write_line("")
            else:
                # Generate timestamp and add to the beginning of the line
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._write_line(f"[{timestamp}] {line}")
        
        # Handle remaining content without newline (wait for next write or flush)
        if self.buffer:
            pass  # Do nothing for now, wait for next write or flush

    def _write_line(self, line):
        # Actual write operation (with newline)
        self.file.write(line + '\n')
        original_stdout.write(line + '\n')

    def flush(self):
        # Handle remaining content in buffer (not yet newline-terminated)
        if self.buffer.strip() != "":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_line(f"[{timestamp}] {self.buffer}")
        self.buffer = ""  # Clear buffer
        self.file.flush()
        original_stdout.flush()