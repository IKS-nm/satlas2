# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))
import satlas2

# -- Project information -----------------------------------------------------

project = "satlas2"
copyright = "2023, IKS"
author = "IKS"

# The full version, including alpha/beta/rc tags
release = satlas2.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
autosummary_generate = False
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "sphinx_toolbox.shields",
]

github_username = "woutergins"
github_repository = "satlas2"

graphviz_output_format = "png"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
}

inheritance_graph_attrs = dict(rankdir="TB")

autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_favicon = "favicon.svg"
html_title = "SATLAS2"
html_logo = "_static/satlas_light.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "repository_url": "https://github.com/iks-nm/satlas2",
    "use_repository_button": True,
    "repository_branch": "master",
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "navigation_depth": 4,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "pygment_light_style": "dracula",
    "pygment_dark_style": "dracula",
    "logo": {
        "text": "SATLAS2",
        "alt_text": "SATLAS2",
        "image_light": "_static/satlas_light.svg",
        "image_dark": "_static/satlas_dark.svg",
    },
}
