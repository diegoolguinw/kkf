"""Sphinx configuration for the KKF documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import kkf  # noqa: E402

project = "KKF"
copyright = "2026, Diego Olguin-Wende"
author = "Diego Olguin-Wende"
release = kkf.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

# myst-parser: let CHANGELOG.md be included as-is (no RST rewrite)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc/autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# numpydoc handles NumPy-style docstrings; napoleon is disabled for parsing
# (kept enabled only so autosummary picks up short summaries consistently).
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",
    "download_all_examples": False,
    "remove_config_comments": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "KKF documentation"
html_theme_options = {
    "github_url": "https://github.com/diegoolguinw/kkf",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "diegoolguinw",
    "github_repo": "kkf",
    "github_version": "main",
    "doc_path": "docs",
}
