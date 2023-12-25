import shutil
import subprocess
import tempfile
from pathlib import Path


class Executor:
    def __init__(self, source: str) -> None:
        self.source = source
        self.temp_dir = None
        self.binary_path = None
        self.source_path = None

    def compile(self) -> bool:
        # Write code and executable in memory.
        self.temp_dir = tempfile.mkdtemp()
        self.source_path = Path(self.temp_dir) / "source.cpp"
        with self.source_path.open("w") as source_file:
            source_file.write(str(self.source))

        self.binary_path = Path(self.temp_dir) / "executable"
        cmd = f"g++ -o {self.binary_path} {self.source_path}"
        try:
            subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            return False

        self.binary_path.chmod(0o755)
        return True

    def exec(self) -> str:
        if self.binary_path is None:
            raise ValueError(
                "Compilation must be successful before executing."
            )

        cmd = str(self.binary_path.resolve())
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir)
