"""Bounded source and reference retrieval for the experiment prompts."""
from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_SOURCE_BYTES = 3 * 1024 * 1024
PATH = re.compile(r'(?:solution|src/[A-Za-z0-9_-]+)\.(?:c|h)\Z')


def parse_object(text: str) -> dict:
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('response does not contain a JSON object')
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError('response must be an object')
    return value


def apply_changes(files: dict[str, str], response: dict) -> tuple[dict[str, str], list[str]]:
    changes = response.get('changes')
    if not isinstance(changes, list) or not 1 <= len(changes) <= 16:
        raise ValueError('expected 1–16 file changes')
    updated = dict(files)
    touched = []
    for change in changes:
        if not isinstance(change, dict) or not isinstance(change.get('path'), str) or not PATH.fullmatch(change['path']):
            raise ValueError('unsafe candidate path')
        path = change['path']
        if path in touched:
            raise ValueError('duplicate candidate path')
        touched.append(path)
        if change.get('delete') is True:
            if path == 'solution.c':
                raise ValueError('solution.c cannot be removed')
            updated.pop(path, None)
        elif isinstance(change.get('content'), str):
            updated[path] = change['content']
        else:
            raise ValueError('change needs content or delete=true')
    if 'solution.c' not in updated or len(updated) > 32 or sum(len(v.encode()) for v in updated.values()) > MAX_SOURCE_BYTES:
        raise ValueError('candidate source limit exceeded')
    return updated, touched


def selected_context(files: dict[str, str], challenge: dict, latest_diff: str = '', failures: str = '',
                     reference_root: Path | None = None, max_chars: int = 12000):
    """Deterministic lexical selection; selected reads are recorded by caller."""
    terms = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]{3,}', json.dumps(challenge).lower()))
    ranked = sorted(files, key=lambda p: (-sum(t in files[p].lower() for t in terms), p))
    sections = ['Source files: ' + ', '.join(sorted(files))]
    reads = []
    for path in ranked:
        room = max_chars - sum(map(len, sections))
        if room < 400:
            break
        content = files[path][:min(room - 100, 6000)]
        sections.append(f'\nFILE {path} lines 1–{len(content.splitlines())}:\n{content}')
        reads.append({'document': path, 'section': f'lines 1–{len(content.splitlines())}',
                      'read_at': datetime.now(timezone.utc).isoformat()})
    if latest_diff:
        sections.append('\nLatest accepted diff:\n' + latest_diff[:1500])
    if failures:
        sections.append('\nRecent Judge failures:\n' + failures[:1200])
    if reference_root:
        doc = reference_root / ('posix.md' if any(x in terms for x in ('http', 'server', 'socket'))
                                else 'c-library.md')
        if doc.exists():
            sections.append('\nLocal reference excerpt:\n' + doc.read_text()[:1000])
            reads.append({'document': str(doc.relative_to(reference_root.parent)), 'section': 'opening',
                          'read_at': datetime.now(timezone.utc).isoformat()})
    return '\n'.join(sections)[:max_chars], reads


def source_diff(before: dict[str, str], after: dict[str, str]) -> str:
    return ''.join(difflib.unified_diff(
        [f'### {p}\n{before[p]}\n' for p in sorted(before)],
        [f'### {p}\n{after[p]}\n' for p in sorted(after)],
        fromfile='accepted', tofile='candidate'))[:2 * 1024 * 1024]
