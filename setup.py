"""
Setup script for SDP-LLM-Agent
Software Defect Prediction using Large Language Models
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]

setup(
    name="sdp-llm-agent",
    version="2.0.0",
    author="SDP-LLM-Agent Contributors",
    description="Software Defect Prediction using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SDP-LLM-Agent",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/SDP-LLM-Agent/issues",
        "Documentation": "https://github.com/yourusername/SDP-LLM-Agent/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/SDP-LLM-Agent",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements-minimal.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "full": read_requirements("requirements.txt"),
    },
    entry_points={
        "console_scripts": [
            "sdp-agent=agent:main",
            "sdp-evaluate=evaluate_agent_v2:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "software defect prediction",
        "llm",
        "code analysis",
        "machine learning",
        "ai agent",
        "defect detection",
        "software quality",
    ],
)
