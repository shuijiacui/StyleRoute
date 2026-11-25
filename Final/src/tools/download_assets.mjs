/** Explicit network-only setup. No downloads occur during model import or app startup. */
import { mkdir, readFile, writeFile, rename, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const exists = p => access(p).then(() => true, () => false);
async function download(url, relative) {
  const destination = path.join(root, relative);
  await mkdir(path.dirname(destination), {recursive:true});
  if (await exists(destination)) {
    const bytes = await readFile(destination);
    console.log(`REUSE ${relative}`);
    return {path:relative,sha256:hash(bytes),bytes:bytes.length,source_url:url};
  }
  let last;
  for(let attempt=0;attempt<3;attempt++) {
    try {
      const response = await fetch(url,{signal:AbortSignal.timeout(240000)});
      if(!response.ok) throw Error(`HTTP ${response.status}: ${url}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      if(bytes.length<100) throw Error(`Empty asset: ${url}`);
      await writeFile(destination+'.part',bytes);
      await rename(destination+'.part',destination);
      console.log(`DOWNLOADED ${relative}: ${bytes.length} bytes`);
      return {path:relative,sha256:hash(bytes),bytes:bytes.length,source_url:url};
    } catch(error) {last=error;console.log(error.message);}
  }
  throw last;
}
const manifestFile = path.join(root,'resources/weights/manifest.json');
await import('./download_design_fonts.mjs');
if (await exists(manifestFile)) {
  const manifest=JSON.parse(await readFile(manifestFile,'utf8'));
  for(const item of [...Object.values(manifest.checkpoints),...manifest.references,manifest.font,manifest.font_license]) {
    const record=await download(item.source_url,item.path);
    if(record.sha256!==item.sha256) throw Error(`SHA256 mismatch: ${item.path}`);
  }
  console.log('All model, source and font assets verified.');
  process.exit(0);
}
const upstream='https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/v0.0.0';
const references=[];
for(const name of ['net.py','function.py','LICENSE']) {
  references.push(await download(`${upstream}/${name}`,`resources/upstream/adain/${name}`));
}
const release='https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0';
const checkpoints={
  encoder:await download(`${release}/vgg_normalised.pth`,'resources/weights/vgg_normalized.pth'),
  decoder:await download(`${release}/decoder.pth`,'resources/weights/decoder.pth'),
};
const font=await download('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf','resources/fonts/NotoSansCJKsc-Regular.otf');
const fontLicense=await download('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE','resources/fonts/OFL.txt');
await writeFile(manifestFile,JSON.stringify({schema_version:1,upstream_repository:'https://github.com/naoto0804/pytorch-AdaIN',upstream_ref:'v0.0.0',upstream_short_commit:'324eede',retrieved_at:new Date().toISOString(),checkpoints,references,font,font_license:fontLicense},null,2)+'\n');
console.log('Asset manifest saved.');
