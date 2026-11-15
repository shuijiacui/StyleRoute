param()
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$imageRoot = Join-Path $projectRoot 'resources/images'
$manifestPath = Join-Path $imageRoot 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allImages = @($manifest.content_images) + @($manifest.style_images)
if ($manifest.content_images.Count -ne 21 -or $manifest.style_images.Count -ne 4) {
    throw 'Expected 21 content photographs and four styles.'
}
$validation = @()
foreach ($item in $allImages) {
    $file = Join-Path $projectRoot $item.local_path
    $decoded = [System.Drawing.Image]::FromFile($file)
    try {
        if ($decoded.RawFormat.Guid -ne [System.Drawing.Imaging.ImageFormat]::Jpeg.Guid) {
            throw "Expected JPEG: $($item.id)"
        }
        if ([Math]::Min($decoded.Width, $decoded.Height) -lt 256) {
            throw "Downloaded image too small: $($item.id)"
        }
        $validation += [ordered]@{
            id = $item.id; local_path = $item.local_path
            width = $decoded.Width; height = $decoded.Height
            format = 'JPEG'; pixel_format = $decoded.PixelFormat.ToString()
            sha256 = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
            decoded = $true
        }
        $item | Add-Member -NotePropertyName width -NotePropertyValue $decoded.Width -Force
        $item | Add-Member -NotePropertyName height -NotePropertyValue $decoded.Height -Force
    } finally {
        $decoded.Dispose()
    }
}

$previewDirectory = Join-Path $imageRoot 'preview'
[System.IO.Directory]::CreateDirectory($previewDirectory) | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 30), $utf8)
$validationReport = [ordered]@{
    checked_at = [DateTime]::UtcNow.ToString('o')
    decoder = 'Windows System.Drawing; full drawing decode also performed in contact sheets'
    image_count = $allImages.Count
    evaluation_content_count = @($manifest.content_images | Where-Object usage -eq 'evaluation').Count
    pilot_content_count = @($manifest.content_images | Where-Object usage -eq 'pilot').Count
    evaluation_pair_count = $manifest.evaluation_pairs.Count
    pilot_pair_count = $manifest.pilot_pairs.Count
    items = $validation
} | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText((Join-Path $imageRoot 'image_validation.json'), $validationReport, $utf8)

$titleFont = [System.Drawing.Font]::new('Arial', 17)
$labelFont = [System.Drawing.Font]::new('Arial', 10)
$smallFont = [System.Drawing.Font]::new('Arial', 9)
$ink = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(35, 43, 52))
$muted = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(90, 100, 110))
$groups = @(
    @{ title = 'BEACH | 1 pilot + 6 evaluation'; items = @($manifest.content_images | Where-Object class_name -eq 'beach') },
    @{ title = 'MOUNTAIN | 1 pilot + 6 evaluation'; items = @($manifest.content_images | Where-Object class_name -eq 'mountain') },
    @{ title = 'DOWNTOWN | 1 pilot + 6 evaluation'; items = @($manifest.content_images | Where-Object class_name -eq 'downtown') },
    @{ title = 'STYLES | Van Gogh / Hokusai / Turner / Degas'; items = @($manifest.style_images) }
)
try {
    foreach ($crop in @($false, $true)) {
        $canvas = [System.Drawing.Bitmap]::new(1400, 1000)
        $graphics = [System.Drawing.Graphics]::FromImage($canvas)
        try {
            $graphics.Clear([System.Drawing.Color]::FromArgb(246, 244, 238))
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $heading = if ($crop) { 'M4 DATASET | content center-crop preview (styles keep aspect ratio)' } else { 'M4 DATASET | downloaded images, original aspect ratios' }
            $graphics.DrawString($heading, $titleFont, $ink, 18, 12)
            for ($row = 0; $row -lt $groups.Count; $row++) {
                $group = $groups[$row]
                $rowY = 53 + 234 * $row
                $graphics.DrawString($group.title, $labelFont, $ink, 18, $rowY)
                for ($column = 0; $column -lt $group.items.Count; $column++) {
                    $item = $group.items[$column]
                    $x = 18 + 196 * $column
                    $y = $rowY + 25
                    $image = [System.Drawing.Image]::FromFile((Join-Path $projectRoot $item.local_path))
                    try {
                        if ($crop -and $row -lt 3) {
                            $side = [Math]::Min($image.Width, $image.Height)
                            $destination = [System.Drawing.Rectangle]::new($x + 8, $y, 170, 170)
                            $graphics.DrawImage($image, $destination, [int](($image.Width - $side) / 2), [int](($image.Height - $side) / 2), $side, $side, [System.Drawing.GraphicsUnit]::Pixel)
                        } else {
                            $scale = [Math]::Min(184.0 / $image.Width, 170.0 / $image.Height)
                            $width = [int]($image.Width * $scale)
                            $height = [int]($image.Height * $scale)
                            $destination = [System.Drawing.Rectangle]::new([int]($x + (184 - $width) / 2), [int]($y + (170 - $height) / 2), $width, $height)
                            $graphics.DrawImage($image, $destination)
                        }
                        $suffix = if ($item.usage -eq 'pilot') { ' [PILOT]' } else { '' }
                        $graphics.DrawString($item.id + $suffix, $labelFont, $ink, $x, $y + 175)
                        $graphics.DrawString("$($image.Width) x $($image.Height)", $smallFont, $muted, $x, $y + 192)
                    } finally {
                        $image.Dispose()
                    }
                }
            }
            $filename = if ($crop) { 'center_crop_contact_sheet.jpg' } else { 'contact_sheet.jpg' }
            $canvas.Save((Join-Path $previewDirectory $filename), [System.Drawing.Imaging.ImageFormat]::Jpeg)
        } finally {
            $graphics.Dispose()
            $canvas.Dispose()
        }
    }
} finally {
    $titleFont.Dispose(); $labelFont.Dispose(); $smallFont.Dispose()
    $ink.Dispose(); $muted.Dispose()
}
Write-Output 'Verified 25 JPEGs and generated both contact sheets.'
