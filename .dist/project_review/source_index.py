from pathlib import Path
import ast,json,re,collections
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
inv=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
print('TOP-LEVEL COUNTS')
for k,v in collections.Counter(f['path'].split('/')[0] for f in inv['files']).items(): print(k,v)
print('\nPYTHON SYNTAX NOTES')
for f in inv['files']:
    if 'syntax_error' in f: print(f['path'],f['syntax_error'])
print('\nROOT MODULES')
for f in inv['files']:
    if '/' not in f['path'] and f['extension']=='.py':
        print(f['path'],f.get('lines'),f.get('classes'),f.get('functions'))
print('\nROUTES')
tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
routes=[]
for n in ast.walk(tree):
    if isinstance(n,ast.FunctionDef):
        for d in n.decorator_list:
            if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr in {'route','get','post','put','delete'}:
                route={'path':ast.literal_eval(d.args[0]),'function':n.name,'line':n.lineno,'methods':['GET']}
                if d.func.attr!='route': route['methods']=[d.func.attr.upper()]
                for kw in d.keywords:
                    if kw.arg=='methods': route['methods']=ast.literal_eval(kw.value)
                routes.append(route)
for r in sorted(routes,key=lambda r:r['line']): print(r)
(OUT/'routes.json').write_text(json.dumps(routes,indent=2),encoding='utf-8')
print('\nRESEARCH SOURCE INDEX')
for f in inv['files']:
    if f['extension']=='.py' and '/' in f['path'] and ('/external/' not in f['path']):
        print(f['path'], 'classes=',f.get('classes',[]),'functions=',f.get('functions',[]),'doc=',(f.get('docstring') or '')[:250].replace('\n',' '))
