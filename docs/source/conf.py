# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GenCoal'
copyright = '2024, Haodong Liu'
author = 'Haodong Liu'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
# os.path.dirname(__file__) 是当前 conf.py 文件的路径。
# '../..' 表示向上两级到项目的根目录。
project_path = os.path.join(os.path.dirname(__file__), '../..')
print("Project path:", project_path)
sys.path.insert(0, project_path)
print("Current sys.path:", sys.path)  # 确认路径打印是否正确

extensions = [
    'sphinx.ext.autodoc',       # Auto-generates docs from docstrings
    'sphinx.ext.napoleon',      # Supports Google/NumPy docstring styles
    'sphinx.ext.viewcode',
    'nbsphinx',                 # For handling Jupyter Notebook files
    'myst_parser',              # For handling Markdown files (if needed)
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
