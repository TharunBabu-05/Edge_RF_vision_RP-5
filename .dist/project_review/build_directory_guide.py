from pathlib import Path
import collections,json
OUT=Path(__file__).resolve().parent
inv=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))

def purpose(d):
    if d=='.dist': return 'Originally empty artifact directory; now contains this generated review.'
    if d.startswith('.git/objects'): return 'Compressed Git object storage and pack metadata, used by version control.'
    if d.startswith('.git/refs'): return 'Git branch, remote, tag, and Codex capture references.'
    if d.startswith('.git/logs'): return 'Git reference-update history (reflogs).'
    if d.startswith('.git/hooks'): return 'Git hook templates; not part of the application runtime.'
    if d.startswith('.git/info'): return 'Local Git repository metadata/exclude rules.'
    if d=='.git': return 'Version-control administration.'
    if d=='models': return 'Main application ONNX model cache; initially contains documentation only.'
    if d=='scene_images': return 'Initially empty destination for captured N/E/S/W camera images.'
    if d=='scene_output': return 'Initially empty destination for reconstructed room JSON.'
    if '__pycache__' in d: return 'Generated Python bytecode cache.'
    if d.startswith('holistic_scene_parsing'):
        if '/osmesa' in d: return 'Legacy research offscreen OpenGL renderer, native build artifacts, and setup.'
        if '/utils' in d: return 'Legacy research PLY geometry I/O support.'
        if '/doc' in d: return 'Research illustration assets.'
        return 'Bundled ECCV 2018 scene parsing research code, separate from the EdgeRF runtime.'
    if d.startswith('DeepPanoContext'):
        if '/external/ldif/gaps/apps/' in d: return 'Native GAPS '+d.rsplit('/',1)[-1]+' command-line geometry/image/point-cloud utility.'
        if '/external/ldif/gaps/pkgs/' in d: return 'GAPS native '+d.rsplit('/',1)[-1]+' library package and supporting source/build files.'
        if '/external/ldif/gaps/vc' in d: return 'GAPS Windows/Visual Studio project, make rules, and GLUT support.'
        if '/external/ldif/gaps/docs' in d: return 'GAPS documentation and package-header reference material.'
        if '/external/ldif/gaps/makefiles' in d: return 'Native GAPS build configuration templates.'
        ext={
            'cooperative_scene_parsing':'SUNRGBD detection evaluation, MATLAB visualization, and dataset metadata',
            'Equirec2Perspec':'Panorama-to-perspective image conversion and example images',
            'frustum_convnet':'SUNRGBD evaluation support',
            'HorizonNet':'Panoramic room-layout estimation and postprocessing',
            'ldif':'Implicit-shape representation, inference, mesh conversion, and native GAPS geometry code',
            'mesh_fusion':'CPU/GPU mesh fusion, marching cubes, offscreen rendering, bindings, builds, and example meshes',
            'panoramic_object_detection':'Panoramic object detection and spherical geometry/overlap support',
            'PIFu':'Implicit-surface geometry and mesh utilities',
            'py360convert':'Cubemap, equirectangular, and perspective conversion with visual examples',
            'pyTorchChamferDistance':'PyTorch native/CUDA Chamfer distance extension'}
        for k,v in ext.items():
            if '/external/'+k in d: return v+'.'
        if '/external' in d: return 'Bundled third-party dependencies for the DPC research pipeline.'
        for k,v in {
            'detector':'2D detector dataset registration and training',
            'ldif':'LIEN/LDIF implicit object reconstruction',
            'mgnet':'Mesh generation, training, testing, and losses',
            'multi_view':'Multi-view embeddings, stitching, merging, and training/testing',
            'pano3d':'Panoramic layout, object boxes, relation/scene graph modeling, training/testing',
            'total3d':'Perspective camera/layout and scene understanding',
            'modules':'Shared neural-network modules'}.items():
            if '/models/'+k in d: return 'DPC research: '+v+'.'
        if '/models' in d: return 'DPC research models, datasets, registries, optimization, metrics, and training/testing infrastructure.'
        if '/demo/input' in d: return 'Example research panorama and serialized input scene.'
        if '/demo/output' in d: return 'Example research detection/reconstruction renderings.'
        if '/demo' in d: return 'DPC reference demo assets.'
        if '/images' in d: return 'Research teaser, animated pipeline illustration, and HDR lighting asset.'
        if '/utils' in d: return 'DPC transforms, geometry, rendering, data preprocessing, relation optimization, and visualization.'
        if '/configs' in d: return 'Research experiment YAML and dataset/configuration helpers.'
        return 'Bundled ICCV 2021 DPC research project; not imported by the root application.'
    return 'See contained file entries in FILE_CATALOGUE.md.'

lines=['**Complete original-directory guide**','','All 174 original directories are listed. Counts refer to the original snapshot and exclude generated review artifacts. Recursive counts include descendants.','', '| Directory | Direct files | Recursive files | Purpose |','|---|---:|---:|---|']
for d in inv['directories']:
    c=checks['directory_groups'][d]
    lines.append(f"| `{d}` | {c['direct_files']} | {c['recursive_files']} | {purpose(d)} |")
(OUT/'DIRECTORY_GUIDE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
counts=collections.Counter(f['extension'] for f in inv['files'])
print('Directories documented:',len(inv['directories']))
print('Root Python line total:',sum(f.get('lines',0) for f in inv['files'] if '/' not in f['path'] and f['extension']=='.py'))
print('Binary/data assets:')
for f in inv['files']:
    if f['extension'] in {'.mat','.pkl','.hdr','.occnet','.ply','.off','.so','.o','.dll','.lib','.json'}:
        print(f['path'],f['bytes'])
