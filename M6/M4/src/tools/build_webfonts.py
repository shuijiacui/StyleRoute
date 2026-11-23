"""Build offline WOFF subsets for the interface; full fonts remain for PNG export."""

from fontTools import subset
from fontTools.ttLib import TTFont

from src.project import project_path


def main():
    characters = set(chr(n) for n in range(32, 256))
    for high in range(0xA1, 0xF8):
        for low in range(0xA1, 0xFF):
            try:
                characters.add(bytes([high, low]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    characters.update(chr(n) for n in range(0x2000, 0x2070))
    characters.update(chr(n) for n in range(0x3000, 0x3040))
    for source, name in (
        ("NotoSerifCJKsc-Regular.otf", "ShanhaiReading"),
        ("MaShanZheng-Regular.ttf", "ShanhaiBrush"),
    ):
        font = TTFont(project_path(f"resources/fonts/{source}"))
        options = subset.Options()
        options.name_IDs = ["*"]  # retain copyright and license records
        builder = subset.Subsetter(options=options)
        selected_characters = characters
        if name == "ShanhaiBrush":
            # Only the fixed brand/hero uses the brush face on screen.
            selected_characters = set("山海来信把风景寄给远方见字如面")
            selected_characters.update(chr(n) for n in range(32, 127))
        builder.populate(text="".join(sorted(selected_characters)))
        builder.subset(font)
        for record in font["name"].names:
            if record.nameID in {1, 2, 3, 4, 6, 16, 17}:
                replacement = "Regular" if record.nameID in {2, 17} else name
                record.string = replacement.encode(record.getEncoding())
        font.flavor = "woff"
        target = project_path(f"resources/fonts/{name}.woff")
        font.save(target)
        print(f"{target.name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
