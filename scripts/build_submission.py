#!/usr/bin/env python3
"""
build_submission.py
===================

One-command bundler that assembles a journal- and bioRxiv-ready
submission package from the manuscript, figures, supplementary tables,
and submission scaffolding in this repository.

Outputs everything under ``submission/build/`` and produces a single
``SUBMISSION_BUNDLE.zip`` for upload.

Idempotent: rerunning safely overwrites the previous build.

Usage
-----
    python scripts/build_submission.py
    python scripts/build_submission.py --repo /path/to/repo
    python scripts/build_submission.py --skip-pdf      # skip pandoc/soffice
    python scripts/build_submission.py --skip-zip      # skip final ZIP

External tools (optional; the script degrades gracefully if missing):
    pandoc, soffice (LibreOffice), img2pdf, PyPDF2/pypdf
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

LOG = logging.getLogger("build_submission")

# --------------------------------------------------------------------------- #
# Paths (relative to repo root)
# --------------------------------------------------------------------------- #

REL_MANUSCRIPT_MD = Path("manuscript/markdown/manuscript_complete.md")
REL_MANUSCRIPT_DOCX = Path("manuscript/CRC_Manuscript_Complete.docx")
REL_SUPP_DOCX = Path("manuscript/Supplementary_Tables.docx")
REL_FIGURES_DIR = Path("manuscript/figures")
REL_DIAGNOSTICS_DIR = Path("figures/diagnostics")
REL_SUPP_CSV_DIR = Path("results/supplementary")
REL_SUBMISSION_DIR = Path("submission")
REL_BUILD_DIR = Path("submission/build")

FIGURE_BASENAMES = [
    "Figure1_Forest_Plot",
    "Figure2_ROC_Curves",
    "Figure3_SHAP_Importance",
    "Figure4_Three_Panel_SHAP",
]

SUBMISSION_SCAFFOLD_FILES = [
    "00_cover_letter.md",
    "01_data_availability.md",
    "02_ethics_statement.md",
    "03_author_contributions.md",
    "04_submission_checklist.md",
    "05_biorxiv_metadata.md",
    "06_reviewer_responses.md",
    "07_pre_submission_qa.md",
]


# --------------------------------------------------------------------------- #
# Manifest tracking
# --------------------------------------------------------------------------- #

@dataclass
class ManifestEntry:
    path: Path           # absolute path
    rel_path: str        # relative to build dir
    size_bytes: int
    sha256: Optional[str] = None
    note: str = ""


@dataclass
class BuildReport:
    repo_root: Path
    build_dir: Path
    entries: List[ManifestEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_utc: str = ""
    finished_utc: str = ""

    def add(self, path: Path, note: str = "", with_hash: bool = False) -> None:
        path = path.resolve()
        if not path.exists():
            self.warnings.append(f"Manifest skip (missing): {path}")
            return
        rel = str(path.relative_to(self.build_dir)) if path.is_relative_to(self.build_dir) else str(path)
        digest = sha256_file(path) if with_hash else None
        self.entries.append(
            ManifestEntry(
                path=path,
                rel_path=rel,
                size_bytes=path.stat().st_size,
                sha256=digest,
                note=note,
            )
        )

    def warn(self, msg: str) -> None:
        LOG.warning(msg)
        self.warnings.append(msg)

    def total_size(self) -> int:
        return sum(e.size_bytes for e in self.entries)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def safe_run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    LOG.info("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Build steps
# --------------------------------------------------------------------------- #

def build_manuscript_pdf(repo: Path, build_dir: Path, report: BuildReport) -> Optional[Path]:
    """Convert manuscript markdown to PDF via pandoc; fall back to soffice on docx."""
    out_pdf = build_dir / "manuscript.pdf"
    md_src = repo / REL_MANUSCRIPT_MD
    docx_src = repo / REL_MANUSCRIPT_DOCX

    if have_tool("pandoc") and md_src.exists():
        cmd = [
            "pandoc",
            str(md_src),
            "-o", str(out_pdf),
            "--from", "markdown",
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "linestretch=1.15",
        ]
        result = safe_run(cmd)
        if result.returncode == 0 and out_pdf.exists():
            LOG.info("Built manuscript.pdf via pandoc")
            return out_pdf
        report.warn(
            f"pandoc failed (rc={result.returncode}); stderr head: "
            f"{result.stderr[:300]!r}"
        )
        # try without xelatex
        cmd2 = [
            "pandoc",
            str(md_src),
            "-o", str(out_pdf),
            "--from", "markdown",
        ]
        result2 = safe_run(cmd2)
        if result2.returncode == 0 and out_pdf.exists():
            LOG.info("Built manuscript.pdf via pandoc (default engine)")
            return out_pdf
        report.warn(f"pandoc default engine also failed: {result2.stderr[:300]!r}")

    if have_tool("soffice") and docx_src.exists():
        cmd = [
            "soffice", "--headless", "--convert-to", "pdf",
            str(docx_src), "--outdir", str(build_dir),
        ]
        result = safe_run(cmd)
        produced = build_dir / (docx_src.stem + ".pdf")
        if result.returncode == 0 and produced.exists():
            if produced != out_pdf:
                shutil.move(str(produced), str(out_pdf))
            LOG.info("Built manuscript.pdf via soffice from %s", docx_src.name)
            return out_pdf
        report.warn(f"soffice conversion failed: rc={result.returncode}; {result.stderr[:300]!r}")

    report.warn(
        "Could not generate manuscript.pdf: neither pandoc nor soffice "
        "produced output. Install pandoc (with a LaTeX engine) or "
        "LibreOffice (`soffice`) and rerun."
    )
    return None


def build_figures_pdf(repo: Path, build_dir: Path, report: BuildReport) -> Optional[Path]:
    """Concatenate Figure1..Figure4 PDFs plus diagnostic PNGs into figures.pdf."""
    out_pdf = build_dir / "figures.pdf"
    fig_dir = repo / REL_FIGURES_DIR
    diag_dir = repo / REL_DIAGNOSTICS_DIR

    fig_pdfs = [fig_dir / f"{n}.pdf" for n in FIGURE_BASENAMES if (fig_dir / f"{n}.pdf").exists()]
    diag_pngs = sorted(diag_dir.glob("*.png")) if diag_dir.exists() else []

    if not fig_pdfs and not diag_pngs:
        report.warn("No figures or diagnostics PNGs found; skipping figures.pdf.")
        return None

    # Try PyPDF2 / pypdf for merging and img2pdf for PNG -> PDF.
    pdf_writer = None
    PdfMerger = None
    try:
        from pypdf import PdfWriter as _PdfWriter
        pdf_writer = _PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfMerger as _PdfMerger
            PdfMerger = _PdfMerger
        except ImportError:
            pass

    try:
        import img2pdf  # type: ignore
        have_img2pdf = True
    except ImportError:
        have_img2pdf = False

    if pdf_writer is None and PdfMerger is None:
        report.warn(
            "Neither pypdf nor PyPDF2 is installed; cannot merge figure PDFs. "
            "Install with `pip install pypdf img2pdf` and rerun. "
            "Falling back to copying figures individually into build/figures/."
        )
        fallback_dir = build_dir / "figures"
        fallback_dir.mkdir(exist_ok=True)
        for p in fig_pdfs + diag_pngs:
            shutil.copy2(p, fallback_dir / p.name)
            report.add(fallback_dir / p.name, note="figure (unmerged)")
        return None

    # Convert PNGs to single PDF first (if img2pdf available),
    # otherwise skip diagnostics from the merged PDF and copy them in.
    png_pdf_path = None
    if diag_pngs and have_img2pdf:
        png_pdf_path = build_dir / "_diagnostics_combined.pdf"
        try:
            with open(png_pdf_path, "wb") as fh:
                fh.write(img2pdf.convert([str(p) for p in diag_pngs]))
        except Exception as exc:  # noqa: BLE001
            report.warn(f"img2pdf failed: {exc}")
            png_pdf_path = None
    elif diag_pngs and not have_img2pdf:
        report.warn(
            "img2pdf not installed; diagnostic PNGs will be copied into "
            "build/figures/diagnostics/ instead of merged into figures.pdf."
        )
        diag_copy_dir = build_dir / "figures" / "diagnostics"
        diag_copy_dir.mkdir(parents=True, exist_ok=True)
        for p in diag_pngs:
            shutil.copy2(p, diag_copy_dir / p.name)
            report.add(diag_copy_dir / p.name, note="diagnostic PNG (unmerged)")

    # Merge PDFs.
    try:
        if pdf_writer is not None:
            writer = pdf_writer()
            for src in fig_pdfs:
                writer.append(str(src))
            if png_pdf_path is not None:
                writer.append(str(png_pdf_path))
            with open(out_pdf, "wb") as fh:
                writer.write(fh)
        else:
            merger = PdfMerger()
            for src in fig_pdfs:
                merger.append(str(src))
            if png_pdf_path is not None:
                merger.append(str(png_pdf_path))
            merger.write(str(out_pdf))
            merger.close()
    except Exception as exc:  # noqa: BLE001
        report.warn(f"PDF merge failed: {exc}")
        return None
    finally:
        if png_pdf_path is not None and png_pdf_path.exists():
            png_pdf_path.unlink()

    LOG.info("Built figures.pdf (%d source PDFs, %d diagnostic PNGs)",
             len(fig_pdfs), len(diag_pngs))
    return out_pdf


def bundle_supplementary(repo: Path, build_dir: Path, report: BuildReport) -> Path:
    """Copy supplementary CSVs + Supplementary_Tables.docx into build/supplementary/."""
    supp_out = build_dir / "supplementary"
    supp_out.mkdir(exist_ok=True)

    csv_dir = repo / REL_SUPP_CSV_DIR
    if csv_dir.exists():
        for csv in sorted(csv_dir.glob("*.csv")):
            dest = supp_out / csv.name
            shutil.copy2(csv, dest)
            report.add(dest, note="supplementary table")
    else:
        report.warn(f"Missing supplementary CSV dir: {csv_dir}")

    docx = repo / REL_SUPP_DOCX
    if docx.exists():
        dest = supp_out / docx.name
        shutil.copy2(docx, dest)
        report.add(dest, note="supplementary tables (Word)")
    else:
        report.warn(f"Missing {docx}")

    # INDEX.csv was already copied above as part of *.csv, but mention it.
    LOG.info("Bundled supplementary into %s", supp_out)
    return supp_out


def bundle_figures_originals(repo: Path, build_dir: Path, report: BuildReport) -> Path:
    """Always copy the main figures (PDF + PNG) into build/figures/main/."""
    out = build_dir / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = repo / REL_FIGURES_DIR
    if not fig_dir.exists():
        report.warn(f"Figure dir missing: {fig_dir}")
        return out
    for name in FIGURE_BASENAMES:
        for ext in ("pdf", "png"):
            src = fig_dir / f"{name}.{ext}"
            if src.exists():
                dest = out / src.name
                shutil.copy2(src, dest)
                report.add(dest, note=f"{name} ({ext.upper()})")
    return out


def bundle_scaffolding(repo: Path, build_dir: Path, report: BuildReport) -> None:
    """Copy submission scaffolding markdown files into the build root."""
    src_dir = repo / REL_SUBMISSION_DIR
    for name in SUBMISSION_SCAFFOLD_FILES:
        src = src_dir / name
        if not src.exists():
            report.warn(f"Submission scaffolding missing: {src}")
            continue
        dest = build_dir / name
        shutil.copy2(src, dest)
        report.add(dest, note="submission scaffolding")


def bundle_manuscript_docx(repo: Path, build_dir: Path, report: BuildReport) -> None:
    """Always include the Word manuscript and markdown source as a fallback."""
    docx = repo / REL_MANUSCRIPT_DOCX
    if docx.exists():
        dest = build_dir / docx.name
        shutil.copy2(docx, dest)
        report.add(dest, note="manuscript (Word)")
    md = repo / REL_MANUSCRIPT_MD
    if md.exists():
        dest = build_dir / "manuscript_complete.md"
        shutil.copy2(md, dest)
        report.add(dest, note="manuscript (Markdown source)")


def write_bundle_readme(build_dir: Path, report: BuildReport) -> Path:
    """Write submission/build/README.md describing the bundle contents."""
    out = build_dir / "README.md"
    md = f"""# Submission Bundle — README

Generated: {report.started_utc}

This directory is auto-generated by `scripts/build_submission.py`. It
contains everything required to submit the manuscript to a journal and
to deposit a pre-print on bioRxiv.

## Contents

| File / Directory | Purpose |
|---|---|
| `manuscript.pdf` | Compiled PDF of the manuscript (pandoc-rendered from `manuscript/markdown/manuscript_complete.md` or fallback `soffice` from the Word source). |
| `manuscript_complete.md` | Plain-text Markdown source of the full manuscript. |
| `CRC_Manuscript_Complete.docx` | Word version of the manuscript (preserves journal-template-friendly formatting). |
| `figures.pdf` | Single PDF containing Figures 1-4 followed by supplementary diagnostic panels. |
| `figures/main/` | Individual main figures (`Figure{{1..4}}.pdf` and `.png`). |
| `figures/diagnostics/` | Diagnostic PNGs (only present if `img2pdf` was unavailable). |
| `supplementary/` | Supplementary Tables S1-S10 (`.csv`), `INDEX.csv`, and `Supplementary_Tables.docx`. |
| `00_cover_letter.md` | Editable cover-letter template (tailor per journal). |
| `01_data_availability.md` | Data Availability statement. |
| `02_ethics_statement.md` | Ethics statement. |
| `03_author_contributions.md` | CRediT-style author contributions + funding + competing interests. |
| `04_submission_checklist.md` | Pre-submission readiness checklist. |
| `05_biorxiv_metadata.md` | bioRxiv submission-form fields. |
| `SUBMISSION_BUNDLE.zip` | All of the above as a single uploadable archive. |

For an authoritative checklist of *what* should be uploaded for any
given target journal see `04_submission_checklist.md` (also at
`submission/04_submission_checklist.md` in the repository root).

## Journal-by-journal upload instructions

### Genome Medicine (BMC / Springer Nature)
1. Title page: paste from `manuscript_complete.md` (heading + author block).
2. Manuscript file: `manuscript.pdf` (or `CRC_Manuscript_Complete.docx` if
   the system insists on Word).
3. Figures: upload `figures/main/Figure{{1..4}}.pdf` individually
   (vector PDF preferred; PNGs available as backup).
4. Supplementary: upload `supplementary/Supplementary_Tables.docx` as the
   main supplement and attach individual `S*.csv` files as additional data.
5. Cover letter: paste `00_cover_letter.md` after substituting `[Editor]`,
   `[Journal Name]`, and the suggested-reviewer block.
6. Statements: paste `01_data_availability.md`, `02_ethics_statement.md`,
   and `03_author_contributions.md` into the corresponding form fields.

### Microbiome (BMC)
Identical workflow to Genome Medicine. Microbiome requires a one-paragraph
plain-language summary; draft from the abstract Background + Conclusion
sentences. Confirm figure files are CMYK-safe.

### mSystems (ASM)
1. ASM submission site accepts a single PDF for initial review — use
   `submission/build/biorxiv_single_pdf.pdf` (produced by
   `scripts/build_biorxiv_pdf.py`).
2. On revision, upload editable Word + individual figures (same as above).
3. Include a 40-word "Importance" statement (draft from abstract Conclusion).

### Gut Microbes (Taylor & Francis)
1. Manuscript file: `CRC_Manuscript_Complete.docx`.
2. Figures: upload individual PDFs from `figures/main/`.
3. Supplementary: zip from `supplementary/` is acceptable.
4. Cover letter: paste `00_cover_letter.md`.

### bioRxiv pre-print
1. Single uploadable PDF: `biorxiv_single_pdf.pdf` (preferred) OR
   `manuscript.pdf` plus separate figure files.
2. Paste metadata fields from `05_biorxiv_metadata.md`.
3. Supplementary ZIP: `SUBMISSION_BUNDLE.zip` or just `supplementary/`.
4. License: CC-BY 4.0.

## Regenerating this bundle

```bash
python scripts/build_submission.py
```

Optional flags:
- `--skip-pdf` — skip PDF compilation (faster for iteration on scaffolding).
- `--skip-zip` — skip producing `SUBMISSION_BUNDLE.zip`.
- `--repo PATH` — point at a non-current repo root.

## Warnings encountered during this build

"""
    if report.warnings:
        md += "\n".join(f"- {w}" for w in report.warnings) + "\n"
    else:
        md += "_None._\n"

    out.write_text(md, encoding="utf-8")
    report.add(out, note="bundle README")
    return out


def write_manifest_md(repo: Path, build_dir: Path, report: BuildReport) -> Path:
    """Write the top-level `submission/MANIFEST.md` with SHA-256 hashes."""
    out = repo / REL_SUBMISSION_DIR / "MANIFEST.md"
    lines: List[str] = []
    lines.append("# Submission Package Manifest")
    lines.append("")
    lines.append(f"Generated: {report.started_utc}")
    lines.append(f"Finished:  {report.finished_utc}")
    lines.append("")
    lines.append(
        "Master index of everything in the submission package. SHA-256 "
        "digests of all final deliverable PDFs and ZIPs are included for "
        "integrity tracking."
    )
    lines.append("")
    lines.append("## `submission/` (scaffolding, human-edited)")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("|---|---|")
    scaffolding_desc = {
        "00_cover_letter.md": "Editable cover-letter template.",
        "01_data_availability.md": "Data Availability statement.",
        "02_ethics_statement.md": "Ethics statement.",
        "03_author_contributions.md": "Author contributions, funding, competing interests.",
        "04_submission_checklist.md": "Pre-submission readiness checklist.",
        "05_biorxiv_metadata.md": "bioRxiv submission-form fields.",
        "06_reviewer_responses.md": "Anticipated reviewer-response prep doc.",
        "07_pre_submission_qa.md": "Internal pre-submission self-critique.",
    }
    for fname, desc in scaffolding_desc.items():
        if (repo / REL_SUBMISSION_DIR / fname).exists():
            lines.append(f"| `submission/{fname}` | {desc} |")
    lines.append("| `submission/MANIFEST.md` | (this file) |")
    lines.append("")

    lines.append("## `submission/build/` (auto-generated)")
    lines.append("")
    lines.append("| Relative path | Size | Note |")
    lines.append("|---|---:|---|")
    for entry in sorted(report.entries, key=lambda e: e.rel_path):
        lines.append(
            f"| `{entry.rel_path}` | {human_size(entry.size_bytes)} | "
            f"{entry.note} |"
        )
    lines.append("")
    lines.append(f"**Total bundle size:** {human_size(report.total_size())}")
    lines.append("")

    deliverables = [e for e in report.entries if e.sha256]
    if deliverables:
        lines.append("## SHA-256 digests (final deliverables)")
        lines.append("")
        lines.append("| File | SHA-256 |")
        lines.append("|---|---|")
        for entry in deliverables:
            lines.append(f"| `{entry.rel_path}` | `{entry.sha256}` |")
        lines.append("")

    if report.warnings:
        lines.append("## Warnings during build")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/build_submission.py")
    lines.append("python scripts/build_biorxiv_pdf.py   # single bioRxiv PDF")
    lines.append("```")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def make_zip(build_dir: Path, report: BuildReport) -> Path:
    """Zip the contents of build_dir (excluding the zip itself) into SUBMISSION_BUNDLE.zip."""
    zip_path = build_dir / "SUBMISSION_BUNDLE.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(build_dir):
            for name in files:
                fp = Path(root) / name
                if fp.resolve() == zip_path.resolve():
                    continue
                arcname = fp.relative_to(build_dir)
                zf.write(fp, arcname=str(arcname))
    return zip_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the CRC metagenomics submission bundle.")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                   help="Repository root (default: parent of scripts/).")
    p.add_argument("--skip-pdf", action="store_true",
                   help="Skip pandoc/soffice manuscript PDF rendering.")
    p.add_argument("--skip-figures", action="store_true",
                   help="Skip merged figures.pdf rendering.")
    p.add_argument("--skip-zip", action="store_true",
                   help="Skip producing SUBMISSION_BUNDLE.zip.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo = args.repo.resolve()
    build_dir = (repo / REL_BUILD_DIR).resolve()
    LOG.info("Repo:      %s", repo)
    LOG.info("Build dir: %s", build_dir)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ensure_clean_dir(build_dir)
    report = BuildReport(repo_root=repo, build_dir=build_dir, started_utc=started)

    # 1. Scaffolding + manuscript source files.
    bundle_scaffolding(repo, build_dir, report)
    bundle_manuscript_docx(repo, build_dir, report)

    # 2. Manuscript PDF.
    if not args.skip_pdf:
        pdf = build_manuscript_pdf(repo, build_dir, report)
        if pdf is not None:
            report.add(pdf, note="manuscript PDF", with_hash=True)
    else:
        report.warn("Skipped manuscript PDF rendering (--skip-pdf).")

    # 3. Figures: copy originals + try to merge into figures.pdf.
    bundle_figures_originals(repo, build_dir, report)
    if not args.skip_figures:
        merged = build_figures_pdf(repo, build_dir, report)
        if merged is not None:
            report.add(merged, note="merged figures PDF", with_hash=True)
    else:
        report.warn("Skipped figures.pdf merging (--skip-figures).")

    # 4. Supplementary bundle.
    bundle_supplementary(repo, build_dir, report)

    # 5. Bundle README.
    write_bundle_readme(build_dir, report)

    # 6. Final ZIP.
    if not args.skip_zip:
        zip_path = make_zip(build_dir, report)
        report.add(zip_path, note="final upload archive", with_hash=True)
    else:
        report.warn("Skipped final ZIP (--skip-zip).")

    report.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 7. Top-level manifest.
    manifest_path = write_manifest_md(repo, build_dir, report)
    LOG.info("Wrote manifest: %s", manifest_path)

    # 8. Print summary.
    print_manifest(report)
    return 0


def print_manifest(report: BuildReport) -> None:
    print()
    print("=" * 72)
    print("SUBMISSION BUNDLE MANIFEST")
    print("=" * 72)
    print(f"Build dir: {report.build_dir}")
    print(f"Started:   {report.started_utc}")
    print(f"Finished:  {report.finished_utc}")
    print()
    print(f"{'Relative path':<55} {'Size':>10}  Note")
    print("-" * 90)
    for entry in sorted(report.entries, key=lambda e: e.rel_path):
        print(f"{entry.rel_path:<55} {human_size(entry.size_bytes):>10}  {entry.note}")
    print("-" * 90)
    print(f"{'TOTAL':<55} {human_size(report.total_size()):>10}")
    print()
    deliverables = [e for e in report.entries if e.sha256]
    if deliverables:
        print("SHA-256 digests (final deliverables):")
        for entry in deliverables:
            print(f"  {entry.sha256}  {entry.rel_path}")
        print()
    if report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
        print()


if __name__ == "__main__":
    sys.exit(main())
