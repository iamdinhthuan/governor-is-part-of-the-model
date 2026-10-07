"""Build the self-contained Overleaf/Elsevier LaTeX source folder from paper/.

Layout (mirrors the earlier NN submission source):
  JSA_Overleaf_Source/
    main_jsa.tex          single file, sections inlined
    main_jsa.bbl          pre-built bibliography (elsarticle-num)
    references.bib
    cas-dc.cls  cas-common.sty  elsarticle-num.bst
    figures/*.pdf
    generated/*.tex       tables produced by make_tables.py

Usage: python3 make_overleaf.py <output_dir>
"""

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = next(d / "paper" for d in (HERE, HERE.parent) if (d / "paper" / "main.tex").exists())
OUT = Path(sys.argv[1]).resolve() / "JSA_Overleaf_Source"

INPUT = re.compile(r"\\input\{(sections|tables)/([^}]+?)(\.tex)?\}")


def inline(text: str, depth: int = 0) -> str:
    def repl(m):
        kind, name = m.group(1), m.group(2)
        if kind == "tables":
            return "\\input{generated/" + name + "}"
        body = (PAPER / "sections" / f"{name}.tex").read_text()
        return f"% ---- {name}.tex ----\n" + inline(body, depth + 1).rstrip("\n")
    assert depth < 5
    return INPUT.sub(repl, text)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "generated").mkdir()

    tex = (PAPER / "main.tex").read_text()
    tex = inline(tex)
    tex = tex.replace("\\graphicspath{{figs/}}", "\\graphicspath{{figures/}}")
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{figs/", r"\\includegraphics\1{", tex)
    tex = tex.replace("\\bibliography{refs}", "\\bibliography{references}")
    # Searchable/copyable text under pdfLaTeX; skipped by XeTeX/LuaTeX engines.
    tex = tex.replace(
        "\\documentclass[a4paper,fleqn]{cas-dc}\n",
        "\\documentclass[a4paper,fleqn]{cas-dc}\n"
        "\\ifdefined\\pdfgentounicode\n  \\input{glyphtounicode}\n  \\pdfgentounicode=1\n\\fi\n",
        1)
    assert "figs/" not in tex and "sections/" not in tex, "unresolved path"
    (OUT / "main_jsa.tex").write_text(tex)

    for f in ("cas-dc.cls", "cas-common.sty", "elsarticle-num.bst"):
        shutil.copy2(PAPER / f, OUT / f)
    shutil.copy2(PAPER / "refs.bib", OUT / "references.bib")

    used_figs = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    for name in sorted(used_figs):
        shutil.copy2(PAPER / "figs" / name, OUT / "figures" / name)
    used_tabs = set(re.findall(r"\\input\{generated/([^}]+)\}", tex))
    for name in sorted(used_tabs):
        shutil.copy2(PAPER / "tables" / f"{name}.tex", OUT / "generated" / f"{name}.tex")
    print(f"wrote {OUT}: {len(used_figs)} figures, {len(used_tabs)} generated tables")


if __name__ == "__main__":
    main()
