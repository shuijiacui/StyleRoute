/** Download the frozen source selection, or verify/restore an existing manifest.
 * Requires Node.js >= 20; no npm packages. Paths resolve from this file.
 * No model evaluation or outcome-based selection is performed here.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const dataRoot = path.join(projectRoot, 'resources/images');
const manifestPath = path.join(dataRoot, 'manifest.json');
const selection = JSON.parse(await readFile(path.join(dataRoot, 'selection.json'), 'utf8'));
const mode = process.argv[2] ?? '--download';
if (!['--download', '--verify', '--restore'].includes(mode)) {
  throw new Error('Usage: node src/tools/download_images.mjs [--download|--verify|--restore]');
}
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const exists = async file => access(file).then(() => true, () => false);
const absolute = relative => {
  const resolved = path.resolve(projectRoot, relative);
  if (!resolved.startsWith(projectRoot + path.sep)) throw new Error('Path outside project');
  return resolved;
};
const plain = html => String(html ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const jsonFile = async (relative, value) => {
  const file = absolute(relative);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(value, null, 2) + '\n', 'utf8');
};

async function fetchBytes(url, accept = '*/*') {
  let error;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'M4CourseProject/1.0 (educational image curation)', Accept: accept },
        signal: AbortSignal.timeout(45000),
      });
      if (!response.ok) {
        const failure = new Error(`HTTP ${response.status}: ${url}`);
        if (response.status === 429) {
          const requested = Number(response.headers.get('retry-after'));
          failure.retryMs = Math.min(60000, Math.max(15000 * (attempt + 1), (requested || 0) * 1000));
        }
        throw failure;
      }
      const buffer = Buffer.from(await response.arrayBuffer());
      if (buffer.length > 30 * 1024 * 1024) throw new Error(`File too large: ${url}`);
      return buffer;
    } catch (caught) {
      error = caught;
      if (attempt < 4) {
        const delay = caught.retryMs ?? 1500 * (attempt + 1);
        console.log(`Retrying after ${delay / 1000}s: ${caught.message}`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  throw error;
}

const fetchJson = async url => JSON.parse((await fetchBytes(url, 'application/json')).toString('utf8'));

async function saveImage(relative, url) {
  const destination = absolute(relative);
  await mkdir(path.dirname(destination), { recursive: true });
  const bytes = await fetchBytes(url, 'image/jpeg');
  if (bytes.length < 1000 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new Error(`Expected a JPEG image, received invalid bytes: ${url}`);
  }
  // Write a temporary file first, so an interrupted download is never a valid asset.
  await writeFile(destination + '.part', bytes);
  await rename(destination + '.part', destination);
  return { sha256: sha256(bytes), file_bytes: bytes.length };
}

async function verify(manifest, restore) {
  if (sha256(await readFile(path.join(dataRoot, 'selection.json'))) !== manifest.selection_sha256) {
    throw new Error('selection.json changed after manifest creation. Create a new dataset version before using it for formal evaluation.');
  }
  const all = [...manifest.content_images, ...manifest.style_images];
  for (const item of all) {
    const file = absolute(item.local_path);
    if (!(await exists(file)) && restore) {
      const result = await saveImage(item.local_path, item.download_url);
      if (result.sha256 !== item.sha256) throw new Error(`Source changed: ${item.id}`);
    }
    if (sha256(await readFile(file)) !== item.sha256) throw new Error(`SHA256 mismatch: ${item.id}`);
    console.log(`VERIFIED ${item.id}`);
  }
  const fingerprints = all.map(item => item.sha256);
  if (new Set(fingerprints).size !== fingerprints.length) throw new Error('Duplicate image bytes');
  const ids = new Set(manifest.content_images.map(item => item.id));
  const styles = new Set(manifest.style_images.map(item => item.id));
  const pilot = new Set(manifest.content_images.filter(item => item.usage === 'pilot').map(item => item.id));
  for (const [name, pairs] of [['pilot_pairs', manifest.pilot_pairs], ['evaluation_pairs', manifest.evaluation_pairs]]) {
    for (const pair of pairs) {
      if (!ids.has(pair.content_id) || !styles.has(pair.style_id)) throw new Error('Unknown pair image');
      if (pilot.has(pair.content_id) !== (name === 'pilot_pairs')) throw new Error('Pilot/evaluation overlap');
    }
  }
  console.log(JSON.stringify({ content: ids.size, styles: styles.size,
    evaluation_pairs: manifest.evaluation_pairs.length, pilot_pairs: manifest.pilot_pairs.length }));
}

if (await exists(manifestPath)) {
  await verify(JSON.parse(await readFile(manifestPath, 'utf8')), mode === '--restore');
  process.exit(0);
}
if (mode !== '--download') throw new Error('No existing manifest. Run with --download first.');

const manifest = {
  schema_version: 1, dataset_id: selection.dataset_id,
  created_at: new Date().toISOString(),
  selection_sha256: sha256(await readFile(path.join(dataRoot, 'selection.json'))),
  content_source: 'Wikimedia Commons; manually curated; NOT Places365',
  selection_policy: selection.selection_policy,
  source_change: 'The proposed Places365 subset was not acquired. Official download page could not be reached in this session. Commons photographs replace it; the original project specification is not modified.',
  preprocessing: { applied: false, note: 'Images retain their downloaded aspect ratios. No local crop, filtering or model preprocessing applied. Commons uses a 1280px-wide thumbnail; Met uses its web-large JPEG. Future content preprocessing must be shared by M4/M5/M6.' },
  content_images: [], style_images: [], evaluation_pairs: [], pilot_pairs: [],
};

// Fetch missing Commons metadata in one batch and retain checkpoints on interruption.
const missing = [];
for (const item of selection.content_images) {
  if (!(await exists(absolute(`resources/images/metadata/commons/${item.id}.json`)))) missing.push(item);
}
if (missing.length) {
  const query = new URL('https://commons.wikimedia.org/w/api.php');
  query.search = new URLSearchParams({ action: 'query', format: 'json', titles: missing.map(item => item.title).join('|'),
    prop: 'imageinfo', iiprop: 'url|size|sha1|timestamp|extmetadata', iiurlwidth: '1280' });
  const batch = await fetchJson(query);
  if (batch.error) throw new Error(JSON.stringify(batch.error));
  for (const item of missing) {
    const page = Object.values(batch.query?.pages ?? {}).find(value => value.title === item.title);
    if (!page?.imageinfo?.length) throw new Error(`Missing Commons file: ${item.title}`);
    await jsonFile(`resources/images/metadata/commons/${item.id}.json`, {
      batchcomplete: batch.batchcomplete, query: { pages: { [page.pageid]: page } },
    });
  }
}

// The list is pinned in selection.json; search ranking is never used when restoring.
for (const item of selection.content_images) {
  const snapshot = JSON.parse(await readFile(absolute(`resources/images/metadata/commons/${item.id}.json`), 'utf8'));
  if (snapshot.error) throw new Error(JSON.stringify(snapshot.error));
  const pages = Object.values(snapshot.query?.pages ?? {});
  const page = pages.find(value => value.imageinfo?.length);
  if (!page) throw new Error(`No Commons image: ${item.title}`);
  const image = page.imageinfo[0];
  const meta = image.extmetadata ?? {};
  const license = meta.LicenseShortName?.value ?? '';
  if (!/^(CC BY(?:-SA)? [\d.]+(?: [a-z]+)?|CC0|Public domain)$/i.test(license)) {
    throw new Error(`License requires manual review: ${item.title}: ${license}`);
  }
  if (image.width < 512 || image.height < 512) throw new Error(`Source too small: ${item.title}`);
  const local = `resources/images/content/${item.usage}/${item.class_name}/${item.id}.jpg`;
  const metadata = `resources/images/metadata/commons/${item.id}.json`;
  const url = image.thumburl ?? image.url;
  const saved = await exists(absolute(local))
    ? await readFile(absolute(local)).then(bytes => ({ sha256: sha256(bytes), file_bytes: bytes.length }))
    : await saveImage(local, url);
  await jsonFile(metadata, snapshot);
  manifest.content_images.push({ ...item, source: 'Wikimedia Commons', local_path: local,
    source_page_id: page.pageid, object_url: image.descriptionurl,
    image_url: image.url, download_url: url, source_sha1: image.sha1,
    source_file_timestamp: image.timestamp,
    width: image.thumbwidth ?? image.width, height: image.thumbheight ?? image.height,
    source_width: image.width, source_height: image.height,
    artist: plain(meta.Artist?.value), artist_html: meta.Artist?.value ?? '',
    credit_html: meta.Credit?.value ?? '', description_html: meta.ImageDescription?.value ?? '',
    license, license_url: meta.LicenseUrl?.value ?? '',
    attribution_required: meta.AttributionRequired?.value ?? null,
    copyrighted: meta.Copyrighted?.value ?? null,
    metadata_snapshot: metadata, retrieved_at: new Date().toISOString(), ...saved });
  console.log(`READY ${item.id} (${saved.file_bytes} bytes)`);
  await new Promise(resolve => setTimeout(resolve, 700));
}

for (const item of selection.style_images) {
  const apiUrl = `https://collectionapi.metmuseum.org/public/collection/v1/objects/${item.object_id}`;
  const metadata = await fetchJson(apiUrl);
  if (!metadata.isPublicDomain || !metadata.primaryImage) throw new Error(`No verified OA image: ${item.object_id}`);
  const local = `resources/images/styles/${item.id}.jpg`;
  const snapshot = `resources/images/metadata/met/${item.id}.json`;
  const url = metadata.primaryImageSmall || metadata.primaryImage;
  const saved = await saveImage(local, url);
  await jsonFile(snapshot, metadata);
  manifest.style_images.push({ ...item, source: 'The Metropolitan Museum of Art',
    title: metadata.title, artist: metadata.artistDisplayName, date: metadata.objectDate,
    object_url: metadata.objectURL || `https://www.metmuseum.org/art/collection/search/${item.object_id}`,
    image_url: metadata.primaryImage, download_url: url, api_url: apiUrl,
    local_path: local, is_public_domain: true, license: 'CC0',
    license_url: 'https://creativecommons.org/publicdomain/zero/1.0/',
    policy_url: 'https://www.metmuseum.org/hubs/open-access', credit: metadata.creditLine,
    metadata_snapshot: snapshot, retrieved_at: new Date().toISOString(), ...saved });
  console.log(`DOWNLOADED ${item.id} (${saved.file_bytes} bytes)`);
}

for (const content of manifest.content_images) {
  for (const style of manifest.style_images) {
    const pair = { pair_id: `${content.id}__${style.id}`, content_id: content.id, style_id: style.id };
    manifest[content.usage === 'pilot' ? 'pilot_pairs' : 'evaluation_pairs'].push(pair);
  }
}
await jsonFile('resources/images/manifest.json', manifest);
await verify(manifest, false);
