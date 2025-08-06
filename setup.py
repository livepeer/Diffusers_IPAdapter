from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Minimal requirements - let parent project (StreamDiffusion) handle the heavy dependencies
install_requires = [
    "torch",  # Core dependency
    "Pillow",  # For image processing
    "numpy",   # Basic array operations
]

setup(
    name="diffusers-ipadapter",
    version="0.1.1",
    description="IPAdapter implementation for HuggingFace Diffusers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="livepeer",
    author_email="",
    url="https://github.com/livepeer/Diffusers_IPAdapter",
    packages=["diffusers_ipadapter", "diffusers_ipadapter.ip_adapter"],
    package_dir={"diffusers_ipadapter": "."},
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    license="MIT",
    keywords="diffusers, stable-diffusion, ip-adapter, image-to-image, ai, machine-learning",
)