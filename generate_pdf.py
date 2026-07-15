#!/usr/bin/env python3
"""Generate RAGBOOK.pdf from RAGBOOK.md using WeasyPrint with correct typography."""

import sys
import markdown
import weasyprint

CSS = """
@page {
    size: A4;
    margin: 75.6pt 62.7pt 60pt 62.7pt;
    @bottom-center {
        content: counter(page);
        font-family: "DejaVu Serif", serif;
        font-size: 10pt;
    }
}

body {
    font-family: "DejaVu Serif", serif;
    font-size: 11pt;
    font-weight: normal;
    line-height: 1.4;
    color: #000000;
    text-align: justify;
    hyphens: auto;
}

h1 {
    font-family: "DejaVu Serif", serif;
    font-size: 22pt;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h2 {
    font-family: "DejaVu Serif", serif;
    font-size: 16pt;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}

h3 {
    font-family: "DejaVu Serif", serif;
    font-size: 13pt;
    font-weight: bold;
    margin-top: 1em;
    margin-bottom: 0.3em;
}

h4, h5, h6 {
    font-family: "DejaVu Serif", serif;
    font-size: 11pt;
    font-weight: bold;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
}

p {
    font-family: "DejaVu Serif", serif;
    font-size: 11pt;
    font-weight: normal;
    margin-top: 0.4em;
    margin-bottom: 0.4em;
}

strong, b {
    font-family: "DejaVu Serif", serif;
    font-weight: bold;
}

em, i {
    font-family: "DejaVu Serif", serif;
    font-style: italic;
}

code {
    font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
    font-size: 9.5pt;
    font-weight: normal;
    background-color: #f5f5f5;
    padding: 0 2pt;
}

pre {
    font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
    font-size: 9.5pt;
    font-weight: normal;
    background-color: #f5f5f5;
    padding: 8pt;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 0.5em 0;
}

pre code {
    font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
    font-size: 9.5pt;
    font-weight: normal;
    background-color: transparent;
    padding: 0;
}

ul, ol {
    margin-top: 0.3em;
    margin-bottom: 0.3em;
    padding-left: 1.5em;
}

li {
    font-family: "DejaVu Serif", serif;
    font-size: 11pt;
    font-weight: normal;
    margin-bottom: 0.2em;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 10pt;
}

th, td {
    border: 1pt solid #cccccc;
    padding: 4pt 6pt;
    text-align: left;
    font-family: "DejaVu Serif", serif;
}

th {
    font-weight: bold;
    background-color: #f0f0f0;
}

hr {
    border: none;
    border-top: 1pt solid #cccccc;
    margin: 1em 0;
}

blockquote {
    margin: 0.5em 0;
    padding-left: 1em;
    border-left: 3pt solid #cccccc;
    font-style: italic;
    color: #444444;
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Construire un Système RAG Production Ready</title>
</head>
<body>
{content}
</body>
</html>
"""


def generate_pdf(md_file: str = "RAGBOOK.md", pdf_file: str = "RAGBOOK.pdf") -> None:
    print(f"Reading {md_file}...")
    with open(md_file, encoding="utf-8") as f:
        md_content = f.read()

    print("Converting Markdown to HTML...")
    extensions = [
        "extra",
        "fenced_code",
        "tables",
        "toc",
        "sane_lists",
    ]
    html_content = markdown.markdown(md_content, extensions=extensions)

    full_html = HTML_TEMPLATE.format(content=html_content)

    print(f"Generating {pdf_file} with WeasyPrint...")
    css = weasyprint.CSS(string=CSS)
    html = weasyprint.HTML(string=full_html)
    html.write_pdf(pdf_file, stylesheets=[css])
    print(f"Done: {pdf_file}")


if __name__ == "__main__":
    md_file = sys.argv[1] if len(sys.argv) > 1 else "RAGBOOK.md"
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else "RAGBOOK.pdf"
    generate_pdf(md_file, pdf_file)
