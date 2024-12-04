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
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_gallery.load_style',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', '**.ipynb_checkpoints',   '404.rst' ]#'**.ipynb',]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']