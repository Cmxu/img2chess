"""
img2chess - A Python library for extracting chess boards from images

This is the main package entry point that re-exports from the nested img2chess module.
"""

# Re-export all public symbols from the nested img2chess module
from .img2chess import *

# Also make the nested module available directly
from . import img2chess 