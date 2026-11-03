from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,struct,numpy as np
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
inv=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
sheet=Image.new('RGB',(1500,1200),'#202630'); draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
records=[]
for f in inv['files']:
    p=ROOT/f['path']
    if p.suffix not in {'.off','.ply'}: continue
    if p.suffix=='.off':
        lines=p.read_text().splitlines()
        n,faces,_=map(int,lines[1].split())
        pts=np.array([list(map(float,line.split()[:3])) for line in lines[2:2+n]])
    else:
        b=p.read_bytes(); end=b.index(b'end_header')+len(b'end_header')
        if b[end:end+2]==b'\r\n': end+=2
        else: end+=1
        header=b[:end].decode('ascii'); n=faces=0
        fields=[]; in_vertices=False
        typemap={'float':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4','uint':'<u4'}
        for line in header.splitlines():
            if line.startswith('element vertex'): n=int(line.split()[-1]); in_vertices=True
            elif line.startswith('element face'): faces=int(line.split()[-1]); in_vertices=False
            elif line.startswith('property ') and in_vertices:
                _,t,name=line.split(); fields.append((name,typemap[t]))
        if 'binary_little_endian' in header:
            arr=np.frombuffer(b,dtype=np.dtype(fields),count=n,offset=end)
            pts=np.column_stack([arr[k] for k in ['x','y','z']])
        elif 'ascii' in header:
            pts=np.array([list(map(float,s.split()[:3])) for s in b[end:].decode().splitlines()[:n]])
        else: raise ValueError('Unsupported PLY format: '+f['path'])
    i=len(records); x=(i%3)*500; y=(i//3)*400
    valid=pts[np.isfinite(pts).all(axis=1)]
    rec={'path':f['path'],'vertices':n,'faces':faces,'bounds_min':valid.min(axis=0).tolist(),'bounds_max':valid.max(axis=0).tolist()}
    records.append(rec)
    pts=valid[::max(1,len(valid)//14000)].astype(float)
    pts-=np.mean(pts,axis=0)
    # Fixed oblique projection for an overview; this is a preview, not metric inference.
    q=np.column_stack([pts[:,0]*.85-pts[:,1]*.55,pts[:,2]-.3*pts[:,0]-.25*pts[:,1]])
    q-=q.min(axis=0); scale=330/max(q.max(axis=0).max(),1e-8)
    q*=scale
    for px,py in q: draw.point((x+80+int(px),y+340-int(py)),fill='#61d7bd')
    draw.text((x+10,y+350),p.name,font=font,fill='white')
    draw.text((x+10,y+375),f'{n:,} vertices; {faces:,} faces',font=font,fill='white')
sheet.save(OUT/'geometry_previews.jpg',quality=92)
(OUT/'geometry.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
