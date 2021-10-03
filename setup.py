from setuptools import find_packages, setup

setup(
    name="bisou_scan",
    version="0.0",
    description="Generates sky for BISOU",
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "pysm3",
        "healpy",
        "astropy"
    ],
)
