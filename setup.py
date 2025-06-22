from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="img2chess",
    version="0.1.0",
    author="img2chess",
    author_email="",
    description="A Python library for extracting chess boards from images and splitting them into squares",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/img2chess",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
    },
    keywords="chess computer-vision opencv image-processing board-games",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/img2chess/issues",
        "Source": "https://github.com/yourusername/img2chess",
    },
)