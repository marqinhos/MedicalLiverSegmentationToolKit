# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))



source_suffix = ['.rst', '.md']
templates_path = ['_templates', 'mods']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_patterns = []
# The master toctree document.
master_doc = 'index'



project = 'Medical Liver Segmentation Toolkit'
copyright = '2024, Marcos Fernández González'
author = 'Marcos Fernández González'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Para soporte de Google y NumPy docstrings
    'sphinx.ext.viewcode',  # Para agregar enlaces al código fuente
    'sphinx_rtd_theme',     # Si estás usando el tema Read the Docs
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    "sphinx.ext.intersphinx",
    #"sphinx_inline_tabs",
]


extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}


# -- Options for Markdown files ----------------------------------------------
#

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
#html_theme = "pydata_sphinx_theme"

html_static_path = ['_static']

#html_logo = 'img/logo_mlfompy.png'
#html_favicon = 'img/logo_mlfompy.ico'

html_title = "Medical Liver Segmentation Toolkit"


html_css_files = [
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]


myst_heading_anchors = 3


html_theme_options = {

    "announcement": "🚀 🩻<em>New Release Medical Liver Segmentation Toolkit!</em>🩻 🚀",

    "sidebar_hide_name": False,

    "light_css_variables": {
        "color-brand-primary": "#140062",
        "color-brand-content": "#7C4DFF",
    },

    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/marqinhos/JetRacer_Autonomous_Driving",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],

    "navigation_with_keys": True,


}


html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

templates = {
    'module': 'template_api.md',
}











# -- Options for LaTeX output ------------------------------------------------
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'MLFoMpy.tex', 'MLFoMpy Documentation', 'MODEV', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'MLFoMpy', 'MLFoMpy Documentation', [author], 1),
]

# -- Options for Epub output -------------------------------------------------
# Disable epub mimetype warnings
suppress_warnings = ["epub.unknown_project_files"]


