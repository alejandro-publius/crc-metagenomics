#!/usr/bin/env python3
"""
build_biorxiv_pdf.py
====================

Produce a SINGLE PDF formatted to bioRxiv pre-print requirements:

* Single column
* Embedded figures inline (after their first reference)
* Embedded supplementary tables as final sections

Primary path: pandoc with a generated Markdown that already has figure
``![](path)`` references inline and supplementary tables appended as CSV
fenced into the document.

Fallback paths (if pandoc / a LaTeX engine is unavailable):

* Use ``soffice --headless --convert-to pdf`` on
  ``manuscript/CRC_Manuscript_Complete.docx`` and emit a clear manual-step
  message describing how to staple the figures in afterwards.

Output: ``submission/build/biorxiv_single_pdf.pdf``
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger("build_biorxiv_pdf")

REL_MANUSCRIPT_MD = Path("manuscript/markdown/manuscript_complete.md")
REL_MANUSCRIPT_DOCX = Path("manuscript/CRC_Manuscript_Complete.docx")
REL_FIGURES_DIR = Path("manuscript/figures")
REL_SUPP_CSV_DIR = Path("results/supplementary")
REL_BUILD_DIR = Path("submission/build")

FIGURE_BASENAMES = [
    "Figure1_Forest_Plot",
    "Figure2_ROC_Curves",
    "Figure3_SHAP_Importance",
    "Figure4_Three_Panel_SHAP",
]

FIGURE_CAPTIONS = {
    "Figure1_Forest_Plot":
        "**Figure 1.** Country-aware leave-one-dataset-out (LODO) per-cohort "
        "AUC for the species-only Random Forest classifier across all ten "
        "cohorts. Whiskers are 10,000-iteration cohort-stratified bootstrap "
        "95% CIs.",
    "Figure2_ROC_Curves":
        "**Figure 2.** ROC curves for the species-only Random Forest, the "
        "joint species-plus-pathway Random Forest, and the joint XGBoost "
        "classifier, computed on pooled held-out LODO predictions "
        "(n = 1,339).",
    "Figure3_SHAP_Importance":
        "**Figure 3.** Top 20 SHAP feature importances for the species-only "
        "Random Forest CRC-vs-control classifier.",
    "Figure4_Three_Panel_SHAP":
        "**Figure 4.** Three-panel SHAP importance plot covering the "
        "CRC-vs-control, healthy-vs-adenoma, and adenoma-vs-CRC tasks. "
        "Oral-associated taxa (e.g. *Fusobacterium nucleatum*, "
        "*Peptostreptococcus stomatis*, *Parvimonas micra*, *Gemella "
        "morbillorum*) rank at the top of the adenoma-vs-CRC panel.",
}


# --------------------------------------------------------------------------- #

def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def safe_run(cmd: List[str]) -> subprocess.CompletedProcess:
    LOG.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# Markdown assembly
# --------------------------------------------------------------------------- #

def _figure_block(repo: Path, basename: str) -> str:
    """Build a Markdown block for one figure (PDF preferred, PNG fallback)."""
    fig_pdf = repo / REL_FIGURES_DIR / f"{basename}.pdf"
    fig_png = repo / REL_FIGURES_DIR / f"{basename}.png"
    src = fig_pdf if fig_pdf.exists() else fig_png
    if not src.exists():
        return f"\n*(Figure asset missing: {basename})*\n"
    caption = FIGURE_CAPTIONS.get(basename, f"**{basename}**")
    return (
        f"\n\n![{caption}]({src.as_posix()}){{ width=100% }}\n\n"
        f"{caption}\n\n"
    )


def _csv_to_markdown_table(path: Path, max_rows: int = 50) -> str:
    """Render a CSV as a small Markdown table. Truncate to max_rows."""
    rows: List[List[str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            rows.append(row)
            if len(rows) > max_rows + 1:
                break
    if not rows:
        return f"_(empty: {path.name})_\n"
    header = rows[0]
    body = rows[1:max_rows + 1]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in body:
        # pad/truncate to header length
        r = (r + [""] * len(header))[:len(header)]
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    truncated = len(rows) > len(body) + 1
    if truncated:
        out.append(f"\n_(showing first {max_rows} rows of {path.name})_")
    return "\n".join(out) + "\n"


def _inject_figures_inline(markdown: str, repo: Path) -> str:
    """Insert figure blocks right after the first textual reference to each figure.

    Looks for case-insensitive substrings 'Figure 1' through 'Figure 4' and
    inserts the corresponding figure block after the *paragraph* containing
    the first match. Figures that never get matched are appended after the
    Results section.
    """
    inserted = {n: False for n in FIGURE_BASENAMES}
    paragraphs = markdown.split("\n\n")
    out: List[str] = []
    for para in paragraphs:
        out.append(para)
        for i, base in enumerate(FIGURE_BASENAMES, start=1):
            if inserted[base]:
                continue
            needle = f"Figure {i}"
            if needle.lower() in para.lower():
                out.append(_figure_block(repo, base).strip())
                inserted[base] = True
    # Append any figures we never matched at end of body.
    if not all(inserted.values()):
        out.append("\n\n## Figures (unreferenced)\n")
        for base, done in inserted.items():
            if not done:
                out.append(_figure_block(repo, base))
    return "\n\n".join(out)


def _supplementary_section(repo: Path) -> str:
    """Build a Markdown section embedding every supplementary CSV as a table."""
    csv_dir = repo / REL_SUPP_CSV_DIR
    if not csv_dir.exists():
        return ""
    parts: List[str] = ["\n\n---\n\n# Supplementary Tables\n"]
    for csv_path in sorted(csv_dir.glob("S*.csv")):
        title = csv_path.stem
        parts.append(f"\n## {title}\n")
        parts.append(_csv_to_markdown_table(csv_path))
    index = csv_dir / "INDEX.csv"
    if index.exists():
        parts.append("\n## Supplementary Index\n")
        parts.append(_csv_to_markdown_table(index, max_rows=20))
    return "\n".join(parts)


def assemble_biorxiv_markdown(repo: Path, build_dir: Path) -> Path:
    src = repo / REL_MANUSCRIPT_MD
    if not src.exists():
        raise FileNotFoundError(f"Manuscript markdown missing: {src}")
    body = src.read_text(encoding="utf-8")

    body_with_figs = _inject_figures_inline(body, repo)
    supp = _supplementary_section(repo)

    out_md = build_dir / "biorxiv_single.md"
    header = (
        "---\n"
        'documentclass: article\n'
        "geometry: margin=1in\n"
        "fontsize: 11pt\n"
        "linestretch: 1.25\n"
        "colorlinks: true\n"
        "---\n\n"
    )
    out_md.write_text(header + body_with_figs + "\n" + supp, encoding="utf-8")
    return out_md


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def convert_with_pandoc(md_path: Path, out_pdf: Path) -> bool:
    if not have_tool("pandoc"):
        return False
    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(out_pdf),
        "--from", "markdown",
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linestretch=1.25",
    ]
    result = safe_run(cmd)
    if result.returncode == 0 and out_pdf.exists():
        return True
    LOG.warning("pandoc xelatex failed: %s", result.stderr[:400])
    # Try default engine.
    cmd = ["pandoc", str(md_path), "-o", str(out_pdf), "--from", "markdown"]
    result = safe_run(cmd)
    if result.returncode == 0 and out_pdf.exists():
        return True
    LOG.warning("pandoc default engine failed: %s", result.stderr[:400])
    return False


def convert_with_soffice(repo: Path, build_dir: Path, out_pdf: Path) -> bool:
    if not have_tool("soffice"):
        return False
    docx = repo / REL_MANUSCRIPT_DOCX
    if not docx.exists():
        return False
    cmd = [
        "soffice", "--headless", "--convert-to", "pdf",
        str(docx), "--outdir", str(build_dir),
    ]
    result = safe_run(cmd)
    produced = build_dir / (docx.stem + ".pdf")
    if result.returncode == 0 and produced.exists():
        shutil.move(str(produced), str(out_pdf))
        LOG.warning(
            "soffice fallback produced %s WITHOUT inline figures or "
            "embedded supplementary tables. Manually staple "
            "manuscript/figures/Figure{1..4}.pdf at the relevant "
            "in-text references and append "
            "results/supplementary/S*.csv as supplementary sections "
            "before bioRxiv upload.",
            out_pdf,
        )
        return True
    LOG.warning("soffice fallback failed: %s", result.stderr[:400])
    return False


def write_manual_instructions(out_pdf: Path, build_dir: Path) -> None:
    """Document the manual LibreOffice path when no converter worked."""
    notes = build_dir / "biorxiv_single_pdf_INSTRUCTIONS.md"
    notes.write_text(
        "# Manual bioRxiv single-PDF assembly\n\n"
        f"`{out_pdf.name}` could not be auto-generated on this machine "
        "because neither `pandoc` (with a LaTeX engine) nor `soffice` "
        "(LibreOffice) is installed.\n\n"
        "## Easiest path (LibreOffice)\n\n"
        "1. Install LibreOffice (provides the `soffice` CLI).\n"
        "2. Open `manuscript/CRC_Manuscript_Complete.docx` in LibreOffice "
        "Writer.\n"
        "3. Insert > Image, then add `manuscript/figures/Figure{1..4}.pdf` "
        "(or `.png`) at the paragraph that first references each figure.\n"
        "4. Append the supplementary tables as new sections at the end of "
        "the document by Insert > Sheet From File for each CSV in "
        "`results/supplementary/`.\n"
        "5. File > Export As > Export Directly as PDF, embed all fonts, "
        f"save as `{out_pdf}`.\n\n"
        "## Reproducible path (pandoc)\n\n"
        "```bash\n"
        "brew install pandoc\n"
        "brew install --cask basictex   # or texlive\n"
        "python scripts/build_biorxiv_pdf.py\n"
        "```\n",
        encoding="utf-8",
    )
    LOG.warning("Wrote manual instructions to %s", notes)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the single bioRxiv PDF.")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                   help="Repository root (default: parent of scripts/).")
    p.add_argument("--keep-md", action="store_true",
                   help="Keep the intermediate biorxiv_single.md.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    repo = args.repo.resolve()
    build_dir = (repo / REL_BUILD_DIR).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = build_dir / "biorxiv_single_pdf.pdf"
    LOG.info("Repo:      %s", repo)
    LOG.info("Output:    %s", out_pdf)

    md_path = assemble_biorxiv_markdown(repo, build_dir)
    LOG.info("Wrote intermediate Markdown: %s", md_path)

    ok = convert_with_pandoc(md_path, out_pdf)
    if not ok:
        LOG.warning("Pandoc unavailable or failed; trying soffice fallback...")
        ok = convert_with_soffice(repo, build_dir, out_pdf)
    if not ok:
        LOG.error(
            "Could not produce %s automatically. Writing manual "
            "instructions instead.", out_pdf,
        )
        write_manual_instructions(out_pdf, build_dir)

    if not args.keep_md and md_path.exists() and ok:
        # Keep .md anyway if conversion failed (useful for debugging).
        md_path.unlink()

    if out_pdf.exists():
        size = out_pdf.stat().st_size
        print(f"OK: {out_pdf} ({size:,} bytes)")
        return 0
    print(f"NOT BUILT: {out_pdf} (see warnings above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
