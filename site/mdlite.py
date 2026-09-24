"""Крошечный конвертер Markdown -> HTML без зависимостей.

Поддерживает: заголовки (#), абзацы, списки (-, *, +, 1.) с вложенностью по
отступу, блоки кода ```lang```, инлайн-код, **жирный**, *курсив*, [ссылки](url),
таблицы, цитаты (>), горизонтальную линию (---). Всё остальное выводится как текст.
HTML в исходнике экранируется.
"""
from __future__ import annotations

import html
import re

_H = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_UL = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")


def _link(m: re.Match) -> str:
    text, href = m.group(1), m.group(2).replace('"', "%22")
    extra = ' target="_blank" rel="noopener"' if href.startswith(("http://", "https://")) else ""
    return f'<a href="{href}"{extra}>{text}</a>'


def _fmt(text: str) -> str:
    text = html.escape(text, quote=False)
    text = _LINK.sub(_link, text)
    text = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", text)
    return text


def inline(text: str) -> str:
    """Инлайн-разметка: код защищается от остальных замен."""
    out, pos = [], 0
    for m in _CODE.finditer(text):
        out.append(_fmt(text[pos:m.start()]))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        pos = m.end()
    out.append(_fmt(text[pos:]))
    return "".join(out)


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [inline(c.strip()) for c in s.split("|")]


def _table(header: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{c}</th>" for c in header)
    body = []
    for r in rows:
        r = (r + [""] * len(header))[: len(header)]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr>' + th + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def _list(lines: list[str], i: int, out: list[str]) -> int:
    n = len(lines)
    stack: list[tuple[int, str]] = []  # (indent, tag)

    def close_deeper(indent: int) -> None:
        while stack and stack[-1][0] > indent:
            out.append(f"</li></{stack.pop()[1]}>")

    while i < n:
        line = lines[i]
        if not line.strip():
            # пустая строка внутри списка допустима, если дальше снова пункт
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and (_UL.match(lines[j]) or _OL.match(lines[j])):
                i = j
                continue
            break
        m, tag = _UL.match(line), "ul"
        if not m:
            m, tag = _OL.match(line), "ol"
        if not m:
            break
        indent = len(m.group(1).replace("\t", "    "))
        text = inline(m.group(2))
        if not stack or indent > stack[-1][0]:
            stack.append((indent, tag))
            out.append(f"<{tag}><li>{text}")
        else:
            close_deeper(indent)
            if not stack:
                stack.append((indent, tag))
                out.append(f"<{tag}><li>{text}")
            elif stack[-1][1] != tag:
                out.append(f"</li></{stack.pop()[1]}>")
                stack.append((indent, tag))
                out.append(f"<{tag}><li>{text}")
            else:
                out.append(f"</li><li>{text}")
        i += 1
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return i


def convert(md: str) -> str:
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if para:
            out.append("<p>" + inline(" ".join(s.strip() for s in para)) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        if not stripped:
            flush()
            i += 1
            continue
        m = _H.match(line)
        if m:
            flush()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if _HR.match(line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + convert("\n".join(buf)) + "</blockquote>")
            continue
        if stripped.startswith("|") and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            flush()
            header = _split_row(lines[i])
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            out.append(_table(header, rows))
            continue
        if _UL.match(line) or _OL.match(line):
            flush()
            i = _list(lines, i, out)
            continue
        para.append(line)
        i += 1
    flush()
    return "\n".join(out)
