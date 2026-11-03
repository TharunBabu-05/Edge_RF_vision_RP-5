from pathlib import Path
import json,re
OUT=Path(__file__).resolve().parent
doc=(OUT/'PROJECT_EXPLAINED.md').read_text(encoding='utf-8')
links=re.findall(r'\]\(([^)]+)\)',doc)
missing=[s for s in links if not (OUT/s).exists()]
inv=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
routes=json.loads((OUT/'routes.json').read_text(encoding='utf-8'))
media=json.loads((OUT/'media.json').read_text(encoding='utf-8'))
print('Original inventory:',len(inv['files']),'files;',len(inv['directories']),'directories; read errors:',len(inv['summary']['errors']))
print('HTTP routes:',len(routes))
print('Report local links:',len(links),'; missing:',missing)
print('Video:',media[-1])
assert not missing
assert len(routes)==47
assert len(inv['files'])==1078
