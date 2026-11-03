from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json, sys, subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'dependencies'))
inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
items=[]; records=[]
for f in inventory['files']:
    if Path(f['path']).suffix.lower() not in {'.png','.jpg','.jpeg','.gif'}: continue
    p=ROOT/f['path']
    with Image.open(p) as im:
        rec={'path':f['path'],'size':im.size,'format':im.format,'frames':getattr(im,'n_frames',1)}
        frames=range(rec['frames']) if rec['frames']<2 else sorted(set([0,rec['frames']//4,rec['frames']//2,3*rec['frames']//4,rec['frames']-1]))
        for frame in frames:
            im.seek(frame)
            items.append((f['path']+(f' [frame {frame}]' if rec['frames']>1 else ''),im.convert('RGB').copy()))
        records.append(rec)

def sheets(items,prefix,cols=3,rows=4):
    for start in range(0,len(items),cols*rows):
        batch=items[start:start+cols*rows]
        sheet=Image.new('RGB',(cols*480,rows*330),'#202630')
        draw=ImageDraw.Draw(sheet)
        for i,(label,im) in enumerate(batch):
            x=(i%cols)*480; y=(i//cols)*330
            thumb=ImageOps.contain(im,(472,260))
            sheet.paste(thumb,(x+(480-thumb.width)//2,y))
            # Preserve full asset path in metadata; wrap the visible identifier.
            label=label.replace('DeepPanoContext(github-1)/DeepPanoContext-main/','DPC/').replace('holistic_scene_parsing(github-2)/holistic_scene_parsing-master/','HSP/')
            draw.text((x+5,y+263),'\n'.join(label[j:j+59] for j in range(0,len(label),59)),font=font,fill='white')
        sheet.save(OUT/f'{prefix}_{start//(cols*rows)+1}.jpg',quality=90)
sheets(items,'images')
try:
    import imageio_ffmpeg
    video=ROOT/'3D_WIFI_render_result.mp4'
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24',output_params=['-vf','scale=640:-2'])
    meta=next(reader)
    # Decode the complete video sequentially; retain one frame every five seconds.
    w,h=meta['size']; video_items=[]; count=0
    step=max(1,round(meta['fps']*5))
    last=None
    for count,data in enumerate(reader,1):
        last=data
        if (count-1)%step==0:
            video_items.append((f'Video {((count-1)/meta["fps"]):.1f}s',Image.frombytes('RGB',(w,h),data)))
    if last and (count-1)%step:
        video_items.append((f'Video end {((count-1)/meta["fps"]):.1f}s',Image.frombytes('RGB',(w,h),last)))
    sheets(video_items,'video',cols=3,rows=4)
    records.append({'path':video.name,'metadata':meta,'decoded_frames':count,'retained_frames':len(video_items)})
except Exception as e:
    records.append({'path':'3D_WIFI_render_result.mp4','error':repr(e)})
(OUT/'media.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
