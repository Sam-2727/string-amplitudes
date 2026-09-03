"""Sphinx configuration for the string amplitudes documentation."""

from importlib.metadata import version as distribution_version

from docutils import nodes
from sphinx import addnodes

project = "string amplitudes"
copyright = "2026"
release = distribution_version("string-amplitudes")
version = release

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
napoleon_use_rtype = False

html_theme = "pydata_sphinx_theme"
html_title = "string amplitudes"
html_logo = "_static/images/string_amplitudes_logo.svg"
html_favicon = "_static/images/string_amplitudes_favicon.png"
html_context = {"default_mode": "light"}
html_theme_options = {
    "back_to_top_button": False,
    "external_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Sam-2727/string-amplitudes",
        }
    ],
    "header_links_before_dropdown": 4,
    "logo": {
        "image_light": "_static/images/string_amplitudes_logo.svg",
        "image_dark": "_static/images/string_amplitudes_logo_white.svg",
    },
    "navbar_align": "content",
    "navbar_end": ["sa-theme-toggle", "navbar-icon-links"],
    "secondary_sidebar_items": [],
    "show_toc_level": 2,
}
html_sidebars = {"**": []}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["api_visibility.css", "site.css"]
html_js_files = ["api_visibility.js", "theme_toggle.js"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def _normalize_autodoc_opening_paragraph(app, what, name, obj, options, lines):
    """Remove accidental continuation indentation from opening prose."""
    first_content = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content is None:
        return

    opening = lines[first_content].lstrip()
    if opening.startswith((".. ", ":", "- ", "* ")):
        return

    for index in range(first_content, len(lines)):
        if not lines[index].strip():
            break
        lines[index] = lines[index].lstrip()


def _simplify_api_documentation(app, doctree, docname):
    """Use concise API headings and bulleted argument/return details."""
    for node_type in (addnodes.desc_parameterlist, addnodes.desc_returns):
        for signature_detail in list(doctree.findall(node_type)):
            signature_detail.parent.remove(signature_detail)

    for field in doctree.findall(nodes.field):
        field_name, field_body = field.children
        label = field_name.astext()
        if label == "Parameters":
            field_name.children[:] = [nodes.Text("Arguments")]
        if label not in {"Parameters", "Returns"}:
            continue
        if len(field_body.children) != 1:
            continue
        if isinstance(field_body.children[0], nodes.bullet_list):
            continue

        content = field_body.children[0]
        field_body.remove(content)
        list_item = nodes.list_item()
        list_item += content
        bullet_list = nodes.bullet_list()
        bullet_list += list_item
        field_body += bullet_list


def _name_home_page_in_navigation(app, pagename, templatename, context, doctree):
    """Give the untitled landing page a label in previous-page navigation."""
    if pagename != "get_started":
        return

    previous_page = context.get("prev")
    if previous_page is not None and previous_page.get("link") == "index.html":
        previous_page["title"] = "Home page"


def setup(app):
    """Register documentation-only presentation transforms."""
    app.connect("autodoc-process-docstring", _normalize_autodoc_opening_paragraph)
    app.connect("doctree-resolved", _simplify_api_documentation)
    app.connect("html-page-context", _name_home_page_in_navigation)
