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

# 获取当前 conf.py 文件所在的目录
conf_dir = os.path.dirname(__file__)

# 计算项目根路径（绝对路径）
project_root = os.path.abspath(os.path.join(conf_dir, "../.."))
print("Project root:", project_root)

# 将项目根路径添加到 sys.path 中
sys.path.insert(0, project_root)

# 拼接 coal 文件夹的路径
coal_path = os.path.join(project_root, 'coal')
sys.path.insert(0, coal_path)

# 打印 sys.path 确认路径添加是否成功
print("Current sys.path:", sys.path)

extensions = [
    'sphinx.ext.autodoc',       # Auto-generates docs from docstrings
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',      # Supports Google/NumPy docstring styles
    'sphinx.ext.viewcode',
    'nbsphinx',                 # For handling Jupyter Notebook files
    'myst_parser',              # For handling Markdown files (if needed)
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
