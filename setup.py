# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

from setuptools import setup, find_packages

setup(
    name="shine",
    version="0.1.0",
    description="SHINE: Entropy-Guided Digital Twins for Myocardial Scar Reconstruction",
    author="Dexiang Zong (CVS.AI, NHRIS, NHCS)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "trimesh>=3.20.0",
        "pyyaml>=6.0",
        "nibabel>=5.0.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
