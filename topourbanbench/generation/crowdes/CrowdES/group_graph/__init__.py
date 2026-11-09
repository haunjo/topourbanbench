"""
Group Graph Module for CrowdES

Components:
- GroupGenerator: Emitter prior-based group clustering
"""

from .group_generator import GroupGenerator, print_group_summary

__all__ = [
    'GroupGenerator',
    'print_group_summary',
]
