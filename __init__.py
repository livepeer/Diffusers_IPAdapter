"""
IPAdapter implementation for HuggingFace Diffusers

This package provides an alternative implementation of the IPAdapter models 
for Huggingface Diffusers with enhanced features including:
- Support for multiple input images
- Image weighting capabilities  
- Negative input image support
- Streamlined workflow with unified IPAdapter class
"""

from .ip_adapter.ip_adapter import IPAdapter

__version__ = "0.1.0"
__author__ = "livepeer"
__email__ = ""
__all__ = ["IPAdapter"]