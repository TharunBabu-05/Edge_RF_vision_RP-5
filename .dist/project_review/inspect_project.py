"""Read-only project inspection; generated reports stay beside this script."""
from pathlib import Path
import ast, collections, hashlib, importlib.util, json, struct, sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
files, dirs, errors = [], [], []
sources = []
for p in sorted(ROOT.rglob('*')):
    if OUT == p or OUT in p.parents:
        continue
    rel = p.relative_to(ROOT).as_posix()
    if p.is_dir():
        dirs.append(rel)
        continue
    try:
        b = p.read_bytes()
        rec = {'path': rel, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'extension': p.suffix.lower()}
        text = None
        if b'\0' not in b[:8192]:
            try:
                text = b.decode('utf-8-sig')
            except UnicodeError:
                text = b.decode('cp1252', errors='replace')
        rec['kind'] = 'text' if text is not None else 'binary'
        if text is not None:
            rec['lines'] = len(text.splitlines())
            rec['preview'] = '\n'.join(text.splitlines()[:12])
            if p.suffix == '.py':
                try:
                    tree = ast.parse(text)
                    rec['docstring'] = ast.get_docstring(tree)
                    rec['classes'] = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
                    rec['functions'] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    rec['imports'] = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
                except SyntaxError as e:
                    rec['syntax_error'] = str(e)
            if not rel.startswith('.git/'):
                sources.append((rel, text))
        files.append(rec)
    except Exception as e:
        errors.append({'path': rel, 'error': str(e)})

modules = ['cv2','numpy','PIL','scipy','flask','imageio','imageio_ffmpeg','onnxruntime','matplotlib','psutil']
summary = {'root':str(ROOT), 'files':len(files),'directories':len(dirs),'bytes':sum(f['bytes'] for f in files),
           'kinds':dict(collections.Counter(f['kind'] for f in files)),
           'extensions':dict(collections.Counter(f['extension'] for f in files)),
           'text_lines':sum(f.get('lines',0) for f in files),
           'packages':{n:bool(importlib.util.find_spec(n)) for n in modules}, 'errors':errors}
(OUT/'inventory.json').write_text(json.dumps({'summary':summary,'directories':dirs,'files':files},indent=2),encoding='utf-8')
(OUT/'tree.txt').write_text('\n'.join(['[D] '+p for p in dirs]+['[F] '+f['path'] for f in files]),encoding='utf-8')
catalog=['# Complete file catalogue','', 'Every original file was opened and hashed. Text previews and symbols are indexing aids, not a claim of line-by-line semantic review.','']
for f in files:
    catalog.extend([f"## `{f['path']}`", f"{f['bytes']:,} bytes; {f['kind']}; SHA-256 `{f['sha256']}`",''])
    if f.get('docstring'): catalog.extend([f['docstring'],''])
    if f.get('classes'): catalog.extend(['Classes: '+', '.join(f['classes']),''])
    if f.get('functions'): catalog.extend(['Functions: '+', '.join(f['functions']),''])
    if f.get('syntax_error'): catalog.extend(['Python 3.12 parse note: '+f['syntax_error'],''])
    if f.get('preview') and not f['path'].startswith('.git/'):
        catalog.extend(['```text',f['preview'],'```',''])
(OUT/'FILE_CATALOGUE.md').write_text('\n'.join(catalog),encoding='utf-8')
# Complete source copies are deliberately not emitted, to avoid duplicating private notes or metadata.
print(json.dumps(summary,indent=2))
