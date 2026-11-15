/** Build a local image gallery and attribution list from the actual manifest. */
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const directory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../resources/images');
const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const plain = value => String(value ?? '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const cards = items => items.map(item => `<article>
  <a href="${escape(item.local_path.replace(/^resources\/images\//, ''))}"><img src="${escape(item.local_path.replace(/^resources\/images\//, ''))}" alt="${escape(item.label_zh)}" loading="lazy"></a>
  <div class="body"><span class="badge ${item.usage === 'pilot' ? 'pilot' : ''}">${item.usage === 'pilot' ? '试运行' : item.usage === 'evaluation' ? '正式评价' : '风格参考'}</span>
  <h3>${escape(item.label_zh)}</h3><code>${escape(item.id)}</code>
  <p>${escape(item.reason_zh)}</p>
  <p class="meta">${escape(item.width)} × ${escape(item.height)} · ${escape(item.license)}<br>作者：${escape(plain(item.artist))}</p>
  <a href="${escape(item.object_url)}" target="_blank" rel="noreferrer">原始作品 / 文件页面 ↗</a>
  </div></article>`).join('\n');

const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>M4 项目图片集</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f2ed;color:#253831;font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.65}main{max-width:1320px;margin:auto;padding:42px 24px}header{max-width:880px;margin-bottom:34px}.eyebrow{font-size:13px;letter-spacing:2px;color:#626d62}h1{font-size:38px;line-height:1.3;margin:12px 0}h2{font-size:24px;margin-top:42px}.intro{font-size:17px}.notice{padding:16px 20px;background:#e6ebdf;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:22px}article{background:#fff;border-radius:14px;overflow:hidden}img{width:100%;height:210px;object-fit:contain;background:#e8e7e2;display:block}.crop .content img{object-fit:cover;aspect-ratio:1;height:auto}.body{padding:18px}h3{font-size:17px;margin:9px 0 4px}p{margin:10px 0}.meta,code{font-size:12px;color:#637168}a{color:#2c6954;text-underline-offset:3px}.badge{background:#e5eee8;padding:3px 8px;border-radius:5px;font-size:12px}.badge.pilot{background:#f5e8c9}button{padding:11px 17px;background:#315b49;color:white;border:0;border-radius:8px;cursor:pointer;font:inherit}footer{margin-top:44px;color:#68726b;font-size:13px}
</style></head><body><main><header><div class="eyebrow">M4 · FIXED IMAGE COLLECTION · V1</div><h1>从真实风景到艺术明信片</h1><p class="intro">21 张内容照片 · 4 张艺术参考 · 72 个正式图对 · 12 个试运行图对</p><p class="notice">内容图来自 Wikimedia Commons，是人工整理的替代图片集，<strong>不是 Places365</strong>。每类第一张仅用于试运行，其余六张用于正式评价。风格参考在两组间共享，不构成未见风格测试。所有图片已实际下载，未运行模型评价。</p><button id="toggle">查看内容图的中心裁剪构图</button><p>点击图片打开本地文件；点击来源查看作者与使用条件。裁剪视图只预览构图，不修改下载图片，也不代表已执行模型预处理。</p></header>
${['beach','mountain','downtown'].map((name,index)=>`<section class="content"><h2>${['海滩 · 水面、岩石与植被','山地 · 雪面、森林与岩峰','城市 · 建筑、街道与光照'][index]}</h2><div class="grid">${cards(manifest.content_images.filter(item=>item.class_name===name))}</div></section>`).join('')}
<section><h2>四种风格参考</h2><div class="grid">${cards(manifest.style_images)}</div></section><footer>来源、文件 SHA256、下载地址和用途：manifest.json · 完整来源快照：metadata/ · 使用与实验说明：README.md</footer></main>
<script>document.getElementById('toggle').onclick=function(){const enabled=document.body.classList.toggle('crop');this.textContent=enabled?'查看原始比例':'查看内容图的中心裁剪构图';};</script></body></html>`;
await writeFile(path.join(directory, 'gallery.html'), html, 'utf8');
const attribution = ['# 图片来源与署名', '',
  '本清单记录实际下载文件的作者、来源与许可；完整 HTML 元数据保存在 metadata/。',
  'Commons 内容图使用其服务器生成的等比例 1280 像素宽缩略图，未本地裁剪、滤镜或风格化；The Met 使用官方 web-large 图像。',
  'CC BY 图片按对应版本署名；CC BY-SA 图片还需遵守对应版本的相同方式共享等条件，发布改编结果时检查该图片的许可链接。各文件许可独立，不对整个图片集另行宣称统一 CC0。', '',
  ...[...manifest.content_images, ...manifest.style_images].flatMap(item => [
    `## ${item.id} — ${item.label_zh}`, '',
    `- 作者：${plain(item.artist)}`,
    `- 作品 / 文件名：${item.title}`,
    `- 来源：${item.object_url}`,
    `- 许可：${item.license} — ${item.license_url || '见来源文件页面'}`,
    `- 文件：${item.local_path}`,
    `- SHA256：\`${item.sha256}\``,
    `- 获取时间：${item.retrieved_at}`, '',
  ])].join('\n');
await writeFile(path.join(directory, 'ATTRIBUTION.md'), attribution, 'utf8');
console.log('Generated gallery.html and ATTRIBUTION.md');
