from setuptools import find_packages, setup

setup(
    name="bisou_sky",
    version="0.0",
    description="Generates sky for BISOU",
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "scipy",
        "pysm3",
        "healpy",
        "astropy"
    ],
    include_package_data=True
)
