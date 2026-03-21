from setuptools import setup, find_packages

setup(
    name="subregion-ae",
    version="0.1.0",
    description=(
        "Deep Convolutional AutoEncoder (DCAE) with sign-stabilized latent space "
        "for ocean data assimilation in the Kuroshio Extension region."
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
)
