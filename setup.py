import re

import setuptools

# Extract the version from the init file.
VERSIONFILE = "kondo_ml/__init__.py"
getversion = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M
)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# Configurations
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "mutual_info",
    ],  # Dependencies
    python_requires=">=3.10",  # Minimum Python version
    name="kondo_ml",  # Package name
    version=new_version,  # Version
    author="L.Rücker",  # Author name
    author_email="ruecker.lukas@gmail.com",  # Author mail
    description="Python package for instance selection algorithms",  # Short package description
    long_description=long_description,  # Long package description
    long_description_content_type="text/markdown",
    url="https://github.com/lurue101/instance-selection-for-regression",  # Url to your Git Repo
    download_url="https://github.com/lurue101/instance-selection-for-regression/archive/"
    + new_version
    + ".tar.gz",
    packages=setuptools.find_packages(),  # Searches throughout all dirs for files to include
    include_package_data=True,  # Must be true to include files depicted in MANIFEST.in
    license_files=["LICENSE"],  # License file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
