import subprocess
import os
from pathlib import Path

def convert_notebooks(directory: Path) -> list[Path]:
    """Convert all .ipynb files in the directory to .py scripts, 
       return a list of the created .py file paths."""
    converted = []
    for notebook in directory.iterdir():
        if notebook.suffix == ".ipynb":
            print(f"Converting {notebook.name} to .py...")
            result = subprocess.run(["jupyter", "nbconvert", "--to", "script", str(notebook)])
            if result.returncode == 0:
                py_file = notebook.with_suffix(".py")
                converted.append(py_file)
            else:
                print(f"Error converting {notebook.name}")
    return converted

def generate_requirements(directory: Path):
    """Run pipreqs on the given directory to generate requirements.txt."""
    print(f"Generating requirements.txt with pipreqs in {directory}...")
    result = subprocess.run(["pipreqs", str(directory), "--force"])
    if result.returncode != 0:
        print("Error running pipreqs.")

def cleanup_files(files: list[Path]):
    """Remove temporary files that were created during conversion."""
    for file in files:
        if file.exists():
            print(f"Cleaning up temporary file {file.name}...")
            file.unlink()

def main():
    notebook_dir = Path(__file__).resolve().parent

    # 1. Convert notebooks
    converted_files = convert_notebooks(notebook_dir)
    
    # 2. Run pipreqs if we converted at least one notebook
    if converted_files:
        generate_requirements(notebook_dir)
    else:
        print("No notebooks converted; skipping pipreqs.")

    # 3. Clean up
    cleanup_files(converted_files)

    print("Done!")

if __name__ == "__main__":
    main()
