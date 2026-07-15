#!/usr/bin/env python3
"""Generate RAGBOOK_cover.pdf — front and back covers, black & white."""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

W, H = A4  # 595.28 x 841.89 pt

TITLE      = "Maîtriser le RAG\npour l'Entreprise"
SUBTITLE   = "De l'ingénierie des embeddings\nà l'industrialisation complète"
AUTHOR     = "Loïc Laureote"
EDITION    = "Première édition"
TAG        = "Copiloté par IA"

BACK_DESC = (
    "Les systèmes RAG (Retrieval-Augmented Generation) représentent aujourd'hui "
    "l'une des architectures les plus puissantes du paysage de l'IA applicative. "
    "Ce livre vous guide de la théorie à la production : embeddings, chunking, "
    "reranking, évaluation et industrialisation complète d'un pipeline RAG.\n\n"
    "Construit par un praticien, pour des praticiens."
)

BACK_BIO = (
    "Loïc Laureote est développeur autodidacte, maker et passionné d'intelligence "
    "artificielle. Originaire de Martinique, il a construit et déployé de nombreux "
    "systèmes RAG en production. Il explore également la biologie de synthèse et "
    "l'ingénierie du vivant comme prochain terrain de l'innovation."
)


def _draw_front(c: canvas.Canvas) -> None:
    # Black background
    c.setFillColor(colors.black)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Decorative horizontal rule (top)
    c.setStrokeColor(colors.white)
    c.setLineWidth(1)
    c.line(20 * mm, H - 28 * mm, W - 20 * mm, H - 28 * mm)

    # TAG — small caps style
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 22 * mm, TAG.upper())

    # Decorative horizontal rule (below tag)
    c.setStrokeColor(colors.white)
    c.line(20 * mm, H - 30 * mm, W - 20 * mm, H - 30 * mm)

    # TITLE
    c.setFillColor(colors.white)
    title_lines = TITLE.split("\n")
    y = H - 80 * mm
    c.setFont("Helvetica-Bold", 32)
    for line in title_lines:
        c.drawCentredString(W / 2, y, line)
        y -= 38

    # Separator line under title
    c.setStrokeColor(colors.white)
    c.setLineWidth(0.5)
    c.line(40 * mm, y - 6, W - 40 * mm, y - 6)

    # SUBTITLE
    y -= 20
    c.setFont("Helvetica", 13)
    c.setFillColor(colors.white)
    for line in SUBTITLE.split("\n"):
        c.drawCentredString(W / 2, y, line)
        y -= 18

    # Central decorative block
    box_y = H / 2 - 30 * mm
    box_h = 28 * mm
    c.setFillColor(colors.white)
    c.rect(20 * mm, box_y, W - 40 * mm, box_h, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(
        W / 2,
        box_y + box_h / 2 + 4,
        "Embeddings · Chunking · Reranking",
    )
    c.setFont("Helvetica", 10)
    c.drawCentredString(
        W / 2,
        box_y + box_h / 2 - 10,
        "Évaluation · Industrialisation · Production",
    )

    # AUTHOR
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W / 2, 60 * mm, AUTHOR)

    # EDITION
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, 52 * mm, EDITION)

    # Bottom rule
    c.setStrokeColor(colors.white)
    c.setLineWidth(1)
    c.line(20 * mm, 40 * mm, W - 20 * mm, 40 * mm)

    # Bottom label
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, 33 * mm, "Intelligence Artificielle Appliquée")


def _wrap_text(text: str, font: str, size: int, max_width: float, c: canvas.Canvas) -> list[str]:
    """Wrap text to fit within max_width."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if stringWidth(test, font, size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_back(c: canvas.Canvas) -> None:
    # Black background
    c.setFillColor(colors.black)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    margin = 20 * mm
    text_w = W - 2 * margin

    # Top rule
    c.setStrokeColor(colors.white)
    c.setLineWidth(1)
    c.line(margin, H - 28 * mm, W - margin, H - 28 * mm)

    # "4e de couverture" label
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, H - 22 * mm, "QUATRIÈME DE COUVERTURE")

    c.setLineWidth(0.5)
    c.line(margin, H - 30 * mm, W - margin, H - 30 * mm)

    # Description block
    y = H - 50 * mm
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "À propos du livre")
    y -= 8 * mm

    c.setFont("Helvetica", 10)
    for paragraph in BACK_DESC.split("\n\n"):
        lines = _wrap_text(paragraph, "Helvetica", 10, text_w, c)
        for line in lines:
            c.drawString(margin, y, line)
            y -= 5.5 * mm
        y -= 3 * mm  # paragraph spacing

    # Separator
    y -= 5 * mm
    c.setStrokeColor(colors.white)
    c.setLineWidth(0.5)
    c.line(margin, y, W - margin, y)
    y -= 8 * mm

    # Author bio
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "L'auteur")
    y -= 7 * mm

    c.setFont("Helvetica", 9)
    for line in _wrap_text(BACK_BIO, "Helvetica", 9, text_w, c):
        c.drawString(margin, y, line)
        y -= 5 * mm

    # Bottom section — white band with title
    band_h = 22 * mm
    band_y = 38 * mm
    c.setFillColor(colors.white)
    c.rect(0, band_y, W, band_h, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(W / 2, band_y + band_h / 2 + 2, "Maîtriser le RAG pour l'Entreprise")
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, band_y + band_h / 2 - 8, AUTHOR + " · " + EDITION)

    # Bottom rule
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.white)
    c.setLineWidth(1)
    c.line(margin, 30 * mm, W - margin, 30 * mm)

    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, 23 * mm, "Intelligence Artificielle Appliquée")


def generate_cover(output: str = "RAGBOOK_cover.pdf") -> None:
    c = canvas.Canvas(output, pagesize=A4)

    # Page 1 — Front cover
    _draw_front(c)
    c.showPage()

    # Page 2 — Back cover
    _draw_back(c)
    c.showPage()

    c.save()
    print(f"Cover generated: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate book covers for RAGBOOK.")
    parser.add_argument(
        "output",
        nargs="?",
        default="RAGBOOK_cover.pdf",
        help="Output PDF file (default: RAGBOOK_cover.pdf)",
    )
    args = parser.parse_args()
    generate_cover(args.output)
