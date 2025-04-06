from setuptools import setup, find_packages

setup(
    name="zchunk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-cpp-python",
        "lark",
        "pydantic"
    ],
    author="Zero Entropy AI",
    author_email="your-email@example.com",
    description="A package for chunking and processing text data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zeroentropy-ai/zchunk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 