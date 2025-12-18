from setuptools import setup, find_packages

setup(
    name="jittorMedTB", 
    version="0.1.0",     
    description="A lightweight, high-performance medical image segmentation framework based on Jittor",
    long_description_content_type="text/markdown",
    packages=find_packages(),
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",

    install_requires=[
        "jittor>=1.3.10",
        "nibabel>=5.0.0",
        "numpy",
    ],

    include_package_data=True,
)