from pathlib import Path
import ast, collections, json, re, subprocess
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
inv=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
result={'root_python':[],'javascript':[],'directory_groups':{},'binary_headers':[]}
for p in sorted(ROOT.glob('*.py')):
    try:
        ast.parse(p.read_text(encoding='utf-8'))
        result['root_python'].append({'path':p.name,'status':'parse OK'})
    except Exception as e:
        result['root_python'].append({'path':p.name,'status':str(e)})
htmls=[(p.name,p.read_text(encoding='utf-8')) for p in ROOT.glob('*.html')]
tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name in {'_build_dashboard_html','_build_3dmap_html'}:
        for sub in ast.walk(n):
            if isinstance(sub,ast.Return) and isinstance(sub.value,ast.Constant): htmls.append((n.name,sub.value.value))
for name,html in htmls:
    i=0
    for m in re.finditer(r'<script([^>]*)>(.*?)</script>',html,re.S|re.I):
        attrs,body=m.groups()
        if 'src=' in attrs or 'importmap' in attrs or not body.strip(): continue
        i+=1
        p=OUT/f'{name}_{i}.mjs'
        p.write_text(body,encoding='utf-8')
        c=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
        result['javascript'].append({'source':name,'script':i,'status':c.returncode,'output':c.stderr})
for d in inv['directories']:
    direct=[f for f in inv['files'] if str(Path(f['path']).parent).replace('\\','/')==d]
    nested=[f for f in inv['files'] if f['path'].startswith(d+'/')]
    result['directory_groups'][d]={'direct_files':len(direct),'recursive_files':len(nested),'bytes':sum(f['bytes'] for f in nested)}
for f in inv['files']:
    if f['kind']=='binary' and not f['path'].startswith('.git/'):
        b=(ROOT/f['path']).read_bytes()
        result['binary_headers'].append({'path':f['path'],'bytes':len(b),'header_hex':b[:24].hex(),'printable_header':''.join(chr(x) if 32<=x<127 else '.' for x in b[:128])})
(OUT/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Root Python files:',len(result['root_python']),'; parse errors:',[r for r in result['root_python'] if r['status']!='parse OK'])
print('JavaScript scripts:',json.dumps(result['javascript'],indent=2))
print('Empty directories:',[k for k,v in result['directory_groups'].items() if not v['recursive_files']])
print('Research subfolders:')
for d,c in result['directory_groups'].items():
    if (d.count('/')<=2 and d.startswith('DeepPano')) or d.startswith('holistic_scene'):
        print(d,c)
