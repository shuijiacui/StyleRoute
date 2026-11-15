/** Local, redistributable typography. No network access is needed at app runtime. */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const assets = [
  ['NotoSerifCJKsc-Regular.otf', 'https://raw.githubusercontent.com/notofonts/noto-cjk/Serif2.003/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf'],
  ['NotoSerif-OFL.txt', 'https://raw.githubusercontent.com/notofonts/noto-cjk/Serif2.003/Serif/LICENSE'],
  ['MaShanZheng-Regular.ttf', 'https://raw.githubusercontent.com/google/fonts/main/ofl/mashanzheng/MaShanZheng-Regular.ttf'],
  ['MaShanZheng-OFL.txt', 'https://raw.githubusercontent.com/google/fonts/main/ofl/mashanzheng/OFL.txt'],
];
await mkdir(path.join(root, 'resources/fonts'), { recursive: true });
const manifestPath = path.join(root, 'resources/fonts/design_manifest.json');
let expected = new Map();
try {
  const previous = JSON.parse(await readFile(manifestPath, 'utf8'));
  expected = new Map(previous.assets.map(item => [item.path, item.sha256]));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
const records = [];
for (const [name, url] of assets) {
  const relative = `resources/fonts/${name}`;
  const target = path.join(root, relative);
  let bytes;
  try { bytes = await readFile(target); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const response = await fetch(url, { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw Error(`HTTP ${response.status}: ${url}`);
    bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(target, bytes);
  }
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  if (expected.has(relative) && expected.get(relative) !== sha256) {
    throw Error(`SHA256 mismatch: ${relative}; inspect the source instead of replacing the manifest.`);
  }
  records.push({ path: relative, source_url: url, bytes: bytes.length, sha256 });
  console.log(`${name}: ${bytes.length} bytes`);
}
await writeFile(manifestPath,
  JSON.stringify({ license: 'SIL OFL 1.1', assets: records }, null, 2) + '\n');
