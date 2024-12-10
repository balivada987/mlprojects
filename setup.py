from setuptools import setup, find_packages
from typing import List

HYPE_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements file and returns a list of dependencies."""
    try:
        with open(file_path, "r") as file_obj:
            requirements = [pkg.strip() for pkg in file_obj.readlines()]
            if HYPE_E_DOT in requirements:
                requirements.remove(HYPE_E_DOT)
        return requirements
    except FileNotFoundError:
        raise FileNotFoundError(f"Requirements file not found: {file_path}")

setup(
    name="ml_first_project",
    version="0.0.1",
    author="Srinivas Balivada",
    author_email="balivada987@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
