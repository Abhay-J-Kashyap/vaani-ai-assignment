"""A small LaTeX-to-structured-text converter, just enough for these three papers.

It follows only the \\input files the document really includes, after comments are stripped (so text that was
commented out of the published paper never reaches the index). It expands the papers' own macros with arguments
(H-Net's \\new{...} expands to nothing, so edit-marked text the PDF doesn't print is dropped as well). It numbers
sections, tables and figures the way LaTeX does (appendix sections lettered), and resolves \\ref to those numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

GREEK = {name: ch for name, ch in zip(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta lambda mu nu xi pi rho sigma tau phi varphi chi psi "
    "omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega".split(),
    "αβγδεεζηθλμνξπρστφφχψωΓΔΘΛΞΠΣΦΨΩ")}
SYMBOLS = {"cdot": "·", "times": "×", "approx": "≈", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
           "in": "∈", "sum": "Σ", "prod": "Π", "infty": "∞", "rightarrow": "→", "to": "→", "leftarrow": "←",
           "Rightarrow": "⇒", "downarrow": "↓", "uparrow": "↑", "pm": "±", "sim": "~", "ldots": "…", "dots": "…",
           "cdots": "…", "log": "log", "exp": "exp", "max": "max", "min": "min", "softmax": "softmax",
           "odot": "⊙", "otimes": "⊗", "oplus": "⊕", "mid": "|", "parallel": "‖", "prime": "′", "circ": "∘",
           "checkmark": "✓", "cmark": "✓", "xmark": "✗", "textdagger": "†", "dagger": "†", "S": "§", "%": "%",
           "&": "&", "_": "_", "#": "#", "$": "$", "{": "{", "}": "}", "textasciitilde": "~", "ell": "ℓ",
           "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "langle": "⟨", "rangle": "⟩", "top": "⊤",
           "quad": " ", "qquad": " ", ",": " ", ";": " ", "!": "", " ": " ", "LaTeX": "LaTeX", "TeX": "TeX"}
ACCENTS = {('"', "u"): "ü", ('"', "o"): "ö", ('"', "a"): "ä", ("'", "e"): "é", ("`", "e"): "è", ("'", "a"): "á",
           ("^", "e"): "ê", ("c", "c"): "ç", ("c", "C"): "Ç", ("'", "E"): "É", ("~", "n"): "ñ", ("'", "i"): "í",
           ("'", "o"): "ó", ('"', "U"): "Ü", ('"', "O"): "Ö"}
FLOATS = ("table", "table*", "figure", "figure*", "wraptable", "wrapfigure", "algorithm", "algorithm*")
DROP_ENVS = ("thebibliography", "keywords", "comment")
UNWRAP_ARG = {"textbf", "textit", "emph", "texttt", "textsc", "textrm", "textsf", "underline", "mathbf", "mathrm",
              "mathit", "mathcal", "mathbb", "boldsymbol", "bm", "text", "operatorname", "mbox", "hbox", "textnormal",
              "uline", "textup", "mathsf", "mathtt", "url", "nolinkurl", "textcolor*", "small", "footnotesize",
              "scriptsize", "large", "Large", "best", "second", "boldmath", "textsuperscript", "textsubscript"}
DROP_WITH_ARG = {"label", "vspace", "hspace", "vskip", "hskip", "includegraphics", "bibliographystyle", "bibliography",
                 "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "parencite", "textcite",
                 "nocite", "phantom", "setlength", "addtolength", "setcounter", "addtocounter", "resizebox*",
                 "rule", "renewcommand", "definecolor", "captionsetup", "pagestyle", "thispagestyle", "newpage",
                 "linewidth", "cmidrule", "specialrule", "arrayrulecolor", "rowcolor", "cellcolor", "color",
                 "fontsize", "selectfont", "looseness", "tightcmidrule", "multirowsetup", "adjustbox"}


@dataclass
class Block:
    kind: str                 # abstract | text | table | figure | algorithm
    sec_num: str              # "3.2.1", "A.2", "" (unnumbered)
    sec_title: str
    sec_path: list[str]       # ["3 Model Architecture", "3.2 Attention", "3.2.1 Scaled Dot-Product Attention"]
    appendix: bool
    text: str
    float_num: str = ""       # "Table 3", "Figure 2"
    caption: str = ""
    labels: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- low-level reading
def group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' (or '['). Return the balanced content and the index after the closing bracket."""
    open_, close = ("{", "}") if s[i] == "{" else ("[", "]")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def read_arg(s: str, i: int) -> tuple[str, int]:
    """One mandatory argument: a {group} or a single token."""
    i = skip_ws(s, i)
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return group(s, i)
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


# --------------------------------------------------------------------------- flattening and macros
def flatten(src_dir: Path, main: str) -> str:
    def load(name: str, depth: int) -> str:
        p = src_dir / name
        if not p.suffix:
            p = p.with_suffix(".tex")
        text = strip_comments(p.read_text(encoding="utf-8", errors="replace"))
        if depth > 8:
            return text
        return re.sub(r"\\(?:input|include)\{([^}]+)\}", lambda m: "\n" + load(m.group(1).strip(), depth + 1) + "\n",
                      text)
    return load(main, 0)


MACRO_DEF = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*(?:\{\\([A-Za-z]+)\}|\\([A-Za-z]+))")


def collect_macros(s: str) -> tuple[dict, str]:
    macros, out, i = {}, [], 0
    for m in MACRO_DEF.finditer(s):
        if m.start() < i:
            continue
        name = m.group(1) or m.group(2)
        j = skip_ws(s, m.end())
        nargs, default = 0, None
        if j < len(s) and s[j] == "[":
            n, j = group(s, j)
            nargs = int(n) if n.strip().isdigit() else 0
            j = skip_ws(s, j)
            if j < len(s) and s[j] == "[":
                default, j = group(s, j)
                j = skip_ws(s, j)
        if j < len(s) and s[j] == "{":
            body, j = group(s, j)
        else:
            body, j = "", j
        if name not in ("citep", "citet"):   # keep these as citation commands, removed later
            macros[name] = (nargs, default, body.replace("\\xspace", ""))
        out.append(s[i:m.start()])
        i = j
    out.append(s[i:])
    rest = "".join(out)
    rest = re.sub(r"\\def\\([A-Za-z]+)\{([^{}]*)\}", lambda m: macros.setdefault(m.group(1), (0, None, m.group(2))) and "", rest)
    return macros, rest


def expand_macros(s: str, macros: dict, passes: int = 6) -> str:
    if not macros:
        return s
    pat = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")(?![A-Za-z])")
    for _ in range(passes):
        out, i, changed = [], 0, False
        for m in pat.finditer(s):
            if m.start() < i:
                continue
            nargs, default, body = macros[m.group(1)]
            j = m.end()
            args = []
            if default is not None:
                j2 = skip_ws(s, j)
                if j2 < len(s) and s[j2] == "[":
                    a, j = group(s, j2)
                    args.append(a)
                else:
                    args.append(default)
            while len(args) < nargs:
                a, j = read_arg(s, j)
                args.append(a)
            rep = body
            for k, a in enumerate(args, 1):
                rep = rep.replace(f"#{k}", a)
            if nargs == 0 and j < len(s) and s[j:j + 2] == "{}":
                j += 2
            out.append(s[i:m.start()])
            out.append(rep)
            i, changed = j, True
        out.append(s[i:])
        s = "".join(out)
        if not changed:
            break
    return s


# --------------------------------------------------------------------------- cleaning text
UNITS = {"ms": "ms", "second": "s", "minute": "min", "hour": "h", "hertz": "Hz", "kilo": "k", "mega": "M",
         "giga": "G", "milli": "m", "percent": "%", "byte": "B", "bit": "bit", "kHz": "kHz", "Hz": "Hz", "s": "s"}


def sub_cmd(s: str, name: str, nargs: int, fmt) -> str:
    """Replace \\name{a1}...{an} using balanced braces (handles nesting, e.g. \\frac{QK^T}{\\sqrt{d_k}})."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    out, i = [], 0
    for _ in range(200):
        m = pat.search(s, i)
        if not m:
            break
        j, args = m.end(), []
        for _ in range(nargs):
            j = skip_ws(s, j)
            if j < len(s) and s[j] == "{":
                a, j = group(s, j)
            else:
                a, j = read_arg(s, j)
            args.append(a)
        out.append(s[i:m.start()])
        out.append(fmt(*args))
        i = j
    out.append(s[i:])
    return "".join(out)


def units(u: str) -> str:
    return "".join(UNITS.get(x, x) for x in re.findall(r"\\([A-Za-z]+)", u)) or u


def clean_math(m: str) -> str:
    m = re.sub(r"\\(left|right|big|Big|bigg|Bigg)[lr]?(?![A-Za-z])", "", m)
    for _ in range(3):
        m = sub_cmd(m, "frac", 2, lambda a, b: f"({a})/({b})")
        m = sub_cmd(m, "nicefrac", 2, lambda a, b: f"{a}/{b}")
        m = sub_cmd(m, "sqrt", 1, lambda a: f"sqrt({a})")
    for _ in range(3):
        m = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", m)
    return m


def clean(s: str, refs: dict | None = None) -> str:
    refs = refs or {}

    def ref(kind: str, label: str) -> str:
        num, typ = refs.get(label.strip(), ("?", ""))
        if kind in ("autoref", "cref", "Cref") and typ:
            return f"{typ} {num}"
        return f"({num})" if kind == "eqref" else num

    # siunitx: \SI{160}{\ms} -> "160 ms" (dropping these lost units next to numbers the citation gate checks)
    s = sub_cmd(s, "SI", 2, lambda n, u: f"{n} {units(u)}")
    s = sub_cmd(s, "si", 1, units)
    s = sub_cmd(s, "num", 1, lambda n: n)
    s = re.sub(r"\\looseness\s*=\s*-?\d+", "", s)
    s = re.sub(r"\\hyperref\[[^\]]*\]\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
    s = re.sub(r"\\(ref|autoref|cref|Cref|eqref|pageref)\*?\{([^}]*)\}", lambda m: ref(m.group(1), m.group(2)), s)
    s = re.sub(r"\\footnote(?:mark|text)?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", r" (\1)", s)
    s = re.sub(r"\\footnotemark(\[[^\]]*\])?", "", s)
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)", lambda m: clean_math(m.group(1) or m.group(2) or m.group(3)), s,
               flags=re.S)
    s = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray|split|aligned|array|cases|matrix|pmatrix|bmatrix)\*?\}"
               r"(\{[^}]*\})?", " ", s)
    s = re.sub(r"\\end\{(equation|align|gather|multline|eqnarray|split|aligned|array|cases|matrix|pmatrix|bmatrix)\*?\}",
               " ", s)
    s = clean_math(s)
    s = re.sub(r"\\item(\[[^\]]*\])?", "\n• ", s)
    s = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?|\\end\{[^}]*\}", "\n", s)
    s = re.sub(r"\{\\(?:em|it|bf|sc|tt|rm|small|footnotesize|scriptsize|large|Large)\s+", "{", s)
    # commands whose argument is dropped (before accents, so \citep is never read as the accent \c + "itep")
    s = re.sub(r"\\rule\s*(\[[^\]]*\])?\{[^}]*\}\{[^}]*\}", "", s)
    for _ in range(3):
        s = re.sub(r"\\(" + "|".join(map(re.escape, DROP_WITH_ARG)) + r")\*?(?![A-Za-z])\s*(\[[^\]]*\])*\s*(\{(?:[^{}]|\{[^{}]*\})*\})?",
                   "", s)
    s = re.sub(r"\\f?colorbox\{[^}]*\}(\{[^}]*\})?\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2", s)
    # accents: symbol accents take one letter; the letter accent \c only in the form \c{x} or \c x
    s = re.sub(r"\\([\"'`^~])\s*\{?([A-Za-z])\}?", lambda m: ACCENTS.get((m.group(1), m.group(2)), m.group(2)), s)
    s = re.sub(r"\\c(?:\{([A-Za-z])\}|\s+([A-Za-z])(?![A-Za-z]))",
               lambda m: ACCENTS.get(("c", m.group(1) or m.group(2)), m.group(1) or m.group(2)), s)
    for _ in range(4):
        s = re.sub(r"\\(" + "|".join(map(re.escape, UNWRAP_ARG)) + r")\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2", s)
    s = re.sub(r"\\textcolor\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
    for _ in range(3):   # \multirow nested inside \multicolumn needs more than one pass
        s = re.sub(r"\\(multicolumn|multirow)\{[^}]*\}(?:\{[^}]*\}|\*)\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2", s)
    s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: GREEK.get(m.group(1), SYMBOLS.get(m.group(1), "")), s)
    s = re.sub(r"\\([%&_#${}, ;!])", lambda m: SYMBOLS.get(m.group(1), ""), s)
    s = s.replace("\\\\", "\n").replace("~", " ").replace("---", "—").replace("--", "–")
    s = s.replace("``", '"').replace("''", '"').replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def clean_tabular(body: str, refs: dict) -> str:
    rows = []
    body = re.sub(r"\\(toprule|midrule|bottomrule)(\[[^\]]*\])?", "", body)
    body = re.sub(r"\\(hline|cline\{[^}]*\}|cmidrule(\([^)]*\))?\{[^}]*\}|specialrule\{[^}]*\}\{[^}]*\}\{[^}]*\}|addlinespace(\[[^\]]*\])?)",
                  "", body)
    for raw in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        cells = [clean(c, refs) for c in re.split(r"(?<!\\)&", raw)]
        cells = [c.replace("\n", " ").strip() for c in cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# --------------------------------------------------------------------------- structure
HEAD = re.compile(r"\\(section|subsection|subsubsection)(\*?)\s*(?:\[[^\]]*\])?\s*\{")
ENV = re.compile(r"\\begin\{(" + "|".join(map(re.escape, FLOATS + DROP_ENVS + ("abstract",))) + r")\}")


def parse(src_dir: Path, main: str) -> list[Block]:
    text = flatten(src_dir, main)
    macros, text = collect_macros(text)
    text = expand_macros(text, macros)
    # Moshi's \ensuremathmode expands to "\ifmmode X \else $X$ \fi"; both branches print X, keep one
    text = re.sub(r"\\ifmmode(.*?)\\else.*?\\fi(?![A-Za-z])", lambda m: m.group(1).strip(), text, flags=re.S)
    text = re.sub(r"\\acks\s*\{", r"\\section*{Acknowledgements}{", text)
    body = text.split("\\begin{document}", 1)[-1].split("\\end{document}", 1)[0]
    body = re.sub(r"\\(title|author|date)\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", "", body)
    body = body.replace("\\maketitle", "")

    # Pass 1: walk once to assign numbers to sections/floats/equations and map labels -> (number, type).
    events = tokenize(body)
    refs: dict[str, tuple[str, str]] = {}
    counters = {"sec": [0, 0, 0], "Table": 0, "Figure": 0, "Algorithm": 0, "eq": 0}
    appendix = False
    numbered = []
    last_head = None
    for ev in events:
        if ev["type"] == "head":
            last_head = ev
        if ev["type"] == "appendix":
            appendix, counters["sec"] = True, [0, 0, 0]
        elif ev["type"] == "head":
            if ev["star"]:
                ev["num"] = ""
            else:
                lvl = ev["level"]
                counters["sec"][lvl] += 1
                for k in range(lvl + 1, 3):
                    counters["sec"][k] = 0
                parts = counters["sec"][:lvl + 1]
                first = chr(ord("A") + parts[0] - 1) if appendix else str(parts[0])
                ev["num"] = ".".join([first] + [str(p) for p in parts[1:]])
            for lab in ev["labels"]:
                refs[lab] = (ev["num"], "Appendix" if appendix and ev["level"] == 0 else "Section")
        elif ev["type"] == "float":
            kind = {"table": "Table", "wraptable": "Table", "figure": "Figure", "wrapfigure": "Figure",
                    "algorithm": "Algorithm"}[ev["env"].rstrip("*")]
            if ev["caption"] is not None:
                counters[kind] += 1
                ev["num"] = f"{kind} {counters[kind]}"
                for lab in ev["labels"]:
                    refs[lab] = (str(counters[kind]), kind)
        elif ev["type"] == "text":
            eq_labels = set()
            for env_body in re.findall(r"\\begin\{(?:equation|align)\}(.*?)\\end\{(?:equation|align)\}", ev["raw"], flags=re.S):
                for line in re.split(r"\\\\", env_body):
                    if "\\nonumber" in line or "\\notag" in line:
                        continue
                    counters["eq"] += 1
                    for lab in re.findall(r"\\label\{([^}]*)\}", line):
                        refs[lab] = (str(counters["eq"]), "Equation")
                        eq_labels.add(lab)
            # a \label placed later in a section's text (not right after the heading) still names that section
            for lab in re.findall(r"\\label\{([^}]*)\}", ev["raw"]):
                if lab not in eq_labels and lab not in refs and last_head is not None:
                    refs[lab] = (last_head["num"], "Section")
        numbered.append(ev)

    # Pass 2: emit blocks with cleaned text, refs resolved.
    blocks: list[Block] = []
    path: list[tuple[int, str]] = []
    cur_num, cur_title, appendix = "", "", False
    for ev in numbered:
        if ev["type"] == "appendix":
            appendix = True
        elif ev["type"] == "head":
            title = clean(ev["title"], refs).replace("\n", " ")
            path = [p for p in path if p[0] < ev["level"]] + [(ev["level"], f"{ev['num']} {title}".strip())]
            cur_num, cur_title = ev["num"], title
        elif ev["type"] == "abstract":
            blocks.append(Block("abstract", "", "Abstract", ["Abstract"], False, clean(ev["raw"], refs)))
        elif ev["type"] == "float":
            cap = clean(ev["caption"] or "", refs).replace("\n", " ")
            kind = ev["env"].rstrip("*").replace("wrap", "")
            if kind == "table":
                tabs = re.findall(r"\\begin\{tabular[x*]?\}(?:\{[^}]*\})?(?:\{(?:[^{}]|\{[^{}]*\})*\})?(.*?)\\end\{tabular[x*]?\}",
                                  ev["raw"], flags=re.S)
                content = "\n".join(clean_tabular(t, refs) for t in tabs)
            elif kind == "algorithm":
                content = clean(re.sub(r"\\caption\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", ev["raw"]), refs)
            else:
                content = ""
            blocks.append(Block(kind, cur_num, cur_title, [p[1] for p in path], appendix, content,
                                float_num=ev.get("num", ""), caption=cap, labels=ev["labels"]))
        elif ev["type"] == "text":
            for para in re.split(r"\n\s*\n", ev["raw"]):
                t = clean(para, refs)
                if len(t) >= 20:
                    blocks.append(Block("text", cur_num, cur_title, [p[1] for p in path], appendix, t))
    return blocks


def tokenize(body: str) -> list[dict]:
    """Split the body into heads, floats, abstract, appendix markers and runs of text, in document order."""
    events, i = [], 0
    pat = re.compile(HEAD.pattern + r"|" + ENV.pattern + r"|\\appendix(?![A-Za-z])")
    while True:
        m = pat.search(body, i)
        if not m:
            events.append({"type": "text", "raw": body[i:]})
            break
        if m.start() > i:
            events.append({"type": "text", "raw": body[i:m.start()]})
        tok = m.group(0)
        if tok.startswith("\\appendix"):
            events.append({"type": "appendix"})
            i = m.end()
        elif m.group(1):   # heading
            title, j = group(body, m.end() - 1)
            labels = []
            k = skip_ws(body, j)
            while body.startswith("\\label{", k):
                lab, k = group(body, k + 6)
                labels.append(lab)
                k = skip_ws(body, k)
            events.append({"type": "head", "level": ["section", "subsection", "subsubsection"].index(m.group(1)),
                           "star": bool(m.group(2)), "title": title, "labels": labels})
            i = k
        else:
            env = m.group(3)
            end_tok = f"\\end{{{env}}}"
            end = body.find(end_tok, m.end())
            end = len(body) if end < 0 else end
            raw = body[m.end():end]
            i = end + len(end_tok)
            if env in DROP_ENVS:
                continue
            if env == "abstract":
                events.append({"type": "abstract", "raw": raw})
                continue
            # A heading placed inside a float (Moshi's appendix does this) still starts a section.
            for hm in list(HEAD.finditer(raw)):
                title, j = group(raw, hm.end() - 1)
                events.append({"type": "head", "level": ["section", "subsection", "subsubsection"].index(hm.group(1)),
                               "star": bool(hm.group(2)), "title": title, "labels": []})
            raw = HEAD.sub("{", raw)
            cap = None
            cm = re.search(r"\\caption(?:of\{[^}]*\})?\s*(?:\[[^\]]*\])?\s*\{", raw)
            if cm:
                cap, _ = group(raw, cm.end() - 1)
            labels = re.findall(r"\\label\{([^}]*)\}", raw)
            events.append({"type": "float", "env": env, "raw": raw, "caption": cap, "labels": labels})
    return events
