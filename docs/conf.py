"""Sphinx configuration for the string amplitudes documentation."""

project = "string amplitudes"
copyright = "2026"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "pydata_sphinx_theme"
html_title = "String Amplitudes"
html_theme_options = {
    "github_url": "https://github.com/Sam-2727/string-amplitudes",
    "show_toc_level": 2,
}
html_static_path = ["_static"]
html_css_files = ["api_visibility.css"]
html_js_files = ["api_visibility.js"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
