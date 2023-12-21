import subprocess
import uuid
from pathlib import Path


class Executor:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.binary_path = source.parent / f"executable_{uuid.uuid4()}"

    def compile(self) -> None:
        cmd = f"g++ -o {self.binary_path} {self.source}"
        subprocess.run(cmd, check=True, shell=True)
        subprocess.run(f"chmod +x {self.binary_path}", shell=True)

    def exec(self) -> str:
        result = subprocess.run(
            self.binary_path, capture_output=True, text=True
        )
        return result.stdout
