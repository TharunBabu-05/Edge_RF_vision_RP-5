import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire('C:/Users/tharu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdf-lib/package.json');
const {PDFDocument}=require('C:/Users/tharu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdf-lib/dist/pdf-lib.js');
const folder='C:/Users/tharu/Edge_RF_vision_RP-5/ppt/EdgeRF_3D_Visual_v2';
const files=['01_Problem_Statement.png','02_Proposed_Solution.png','03_Technical_Architecture.png'];
const doc=await PDFDocument.create();
doc.setTitle('EdgeRF 3D — Problem, Solution and Technical Architecture');
doc.setSubject('Three illustrated presentation slides. Concept visuals; current and planned capabilities distinguished.');
doc.setAuthor('EdgeRF 3D');
for(const file of files){
 const png=await doc.embedPng(await fs.readFile(folder+'/'+file));
 const width=960, height=width*png.height/png.width;
 const page=doc.addPage([width,height]);
 page.drawImage(png,{x:0,y:0,width,height});
 console.log(JSON.stringify({file,imageWidth:png.width,imageHeight:png.height,pdfWidth:width,pdfHeight:height}));
}
const output=folder+'/EdgeRF_3D_Visual_Presentation.pdf';
await fs.writeFile(output,await doc.save());
const check=await PDFDocument.load(await fs.readFile(output));
if(check.getPageCount()!==3)throw Error('Expected exactly three PDF pages');
console.log(JSON.stringify({output,pages:check.getPageCount(),bytes:(await fs.stat(output)).size}));
