from pathlib import Path
import subprocess,sys
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'dependencies'))
import imageio_ffmpeg
p=ROOT/'DeepPanoContext(github-1)/DeepPanoContext-main/images/photo_studio_01_2k.hdr'
r=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-i',str(p),'-vf','scale=1024:-1','-frames:v','1','-y',str(OUT/'hdr_preview.png')],capture_output=True,text=True)
print('HDR preview exit:',r.returncode,r.stderr)
