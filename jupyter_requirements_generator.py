import subprocess
import os
from pathlib import Path

def convert_notebooks(directory: Path) -> list[Path]:
    """Convert all Jupyter notebooks in a specified directory to Python
    scripts.

    This function iterates through all files in the given directory,
    checking for files with the `.ipynb` extension. For each notebook found,
    it uses the `jupyter nbconvert` command to convert the notebook into a
    Python script. The paths of the created `.py` files are collected and
    returned as a list. If an error occurs during the conversion process, an
    error message is printed to the console.

    Args:
        directory (Path): The directory containing the Jupyter notebooks to convert.

    Returns:
        list[Path]: A list of paths to the converted Python script files.
    """
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
    """Run pipreqs on the given directory to generate requirements.txt.

    This function executes the pipreqs command in the specified directory to
    automatically generate a requirements.txt file. It prints a message
    indicating the start of the process and checks the result of the
    subprocess execution. If the pipreqs command fails, an error message is
    printed to inform the user.

    Args:
        directory (Path): The directory where pipreqs will be run to
    """
    print(f"Generating requirements.txt with pipreqs in {directory}...")
    result = subprocess.run(["pipreqs", str(directory), "--force"])
    if result.returncode != 0:
        print("Error running pipreqs.")

def cleanup_files(files: list[Path]):
    """Remove temporary files created during conversion.

    This function iterates through a list of file paths and removes each
    file if it exists. It is typically used to clean up temporary files that
    are no longer needed after a conversion process.

    Args:
        files (list[Path]): A list of Path objects representing
    """
    for file in files:
        if file.exists():
            print(f"Cleaning up temporary file {file.name}...")
            file.unlink()

def main():
    """Run the main workflow for converting notebooks and managing
    dependencies.

    This function orchestrates the process of converting Jupyter notebooks
    in the current directory, generating a requirements file if any
    notebooks were converted, and cleaning up any temporary files created
    during the conversion process. It first determines the directory
    containing the notebooks, then calls the necessary functions to perform
    each step of the workflow. If no notebooks are converted, it will skip
    the requirements generation step and notify the user.
    """

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
