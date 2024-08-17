# Configuration file for the Sphinx documentation builder.
#
# To build the xicsrt documentation use the command:
#   sphinx-build -M html doc_source doc_build
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
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'XICSRT'
copyright = '2024, Novimir A. Pablant'
author = 'Novimir A. Pablant'

# The full version, including alpha/beta/rc tags
from xicsrt._version import __version__
release = __version__
version = __version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_rtd_theme',
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Define any custom options.
add_module_names = False
todo_include_todos = True
todo_link_only = True

napoleon_custom_sections = ['Programming Notes', 'Private Keywords']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Any theme specific options
html_theme_options = {
    'display_version': True,
    }

# -- Options for AutoDoc -----------------------------------------------------

#autodoc_default_options = {'member-order': 'bysource'}

from sphinx.ext import autodoc

class MirClassDocumenter(autodoc.ClassDocumenter):
    """
    An extention 'Class' with the following new options:
    - 'nodocstring'
    - 'nosignature'
    """
    objtype = 'mirclass'
    directivetype = 'class'

    option_spec = autodoc.ClassDocumenter.option_spec
    option_spec['nodocstring'] = autodoc.bool_option
    option_spec['nosignature'] = autodoc.bool_option
    option_spec['nopublic'] = autodoc.bool_option

class MirModuleDocumenter(autodoc.ModuleDocumenter):
    """
    An extention 'Module' with the following new options:
    - 'nodocstring'
    """
    objtype = 'mirmodule'
    directivetype = 'module'

    option_spec = autodoc.ModuleDocumenter.option_spec
    option_spec['nodocstring'] = autodoc.bool_option
    option_spec['nopublic'] = autodoc.bool_option

def hide_non_private(app, what, name, obj, skip, options):
    """
    Enables showing only private-member while supressing public-members.
    """

    if what in ['mirmodule', 'mirclass']:
        if 'nopublic' in options:
            # if private-members is set, show only private members
            if 'private-members' in options and not name.startswith('_'):
                # skip public methods
                return True

    # do not modify skip - private methods will be shown
    return None

def hide_docstring(app, what, name, obj, options, lines):
    """
    Enables the 'nodocstring' option.
    """
    if what in ['mirclass', 'mirmodule']:
        if 'nodocstring' in options:
            lines.clear()

def hide_signature(app, what, name, obj, options, sig, anno):
    """
    Enables the 'nodsigniture' option.
    """
    if what in ['mirclass', 'mirmodule']:
        if 'nosignature' in options:
            return '', None
    return sig, anno

def setup(app):
    app.add_autodocumenter(MirClassDocumenter)
    app.add_autodocumenter(MirModuleDocumenter)
    app.connect('autodoc-skip-member', hide_non_private)
    app.connect('autodoc-process-docstring', hide_docstring)
    app.connect('autodoc-process-signature', hide_signature)