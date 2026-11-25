"""Optional real-browser acceptance check; needs Playwright and local Edge/Chromium."""

import argparse
import io
import json

from PIL import Image
from playwright.sync_api import expect, sync_playwright

from src.project import project_path


def check_viewport(page):
    page.mouse.move(10, 10)
    page.mouse.wheel(0, 750)
    page.keyboard.press("PageDown")
    main = page.locator('[data-testid="stMain"]')
    assert main.evaluate("el => el.scrollTop === 0 && window.scrollY === 0")
    assert main.evaluate("""el => [...el.querySelectorAll('button, img, table, [role="tab"]')]
        .filter(node => node.checkVisibility() && node.offsetWidth)
        .every(node => node.getBoundingClientRect().bottom <= window.innerHeight + 1)""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", default="outputs/runs/ui_redesign")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"browser": args.channel, "checks": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, accept_downloads=True)
        page.set_default_timeout(60000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        generate = page.get_by_role("button", name="仅生成 AdaIN", exact=True)
        expect(generate).to_be_visible(timeout=60000)
        page.evaluate("document.fonts.ready")
        references = page.locator(".st-key-reference_view img")
        expect(references).to_have_count(2)
        for picture in references.all():
            picture.evaluate("el => el.decode()")
        expect(page.get_by_role("heading", name="把风景，寄给远方。")).to_be_visible()
        assert page.evaluate("document.fonts.check('16px ShanhaiReading')")
        assert page.evaluate("document.fonts.check('30px ShanhaiBrush')")
        page.screenshot(path=str(output / "home.png"), full_page=True)
        check_viewport(page)
        expect(page.get_by_role("button", name="制作明信片 →", exact=True)).to_be_disabled()
        assert page.get_by_label("明信片标题", exact=True).count() == 0
        assert (
            page.locator(".letter-topbar").evaluate("el => getComputedStyle(el).position")
            == "fixed"
        )
        page.locator('[data-testid="stMain"]').evaluate("el => el.scrollTop = 450")
        assert abs(page.locator(".letter-topbar").bounding_box()["y"]) < 1
        report["checks"].append("fixed header, local fonts, disabled composer before generation")
        generate.click()
        compose = page.get_by_role("button", name="制作明信片 →", exact=True)
        expect(compose).to_be_enabled()
        page.screenshot(path=str(output / "generated.png"), full_page=True)
        compose.click()
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        export = dialog.get_by_role("tab", name="预览与导出")
        edit = dialog.get_by_role("tab", name="编辑信笺")
        export.click()
        expect(dialog.get_by_role("button", name="下载这封来信 PNG", exact=True)).to_be_visible()
        preview = dialog.locator(".st-key-letter_preview img")
        expect(preview).to_be_visible()
        page.screenshot(path=str(output / "dialog.png"), full_page=True)
        before = preview.get_attribute("src")
        edit.click()
        title = dialog.get_by_label("明信片标题", exact=True)
        title.fill("寄自山海之间")
        title.press("Enter")
        export.click()
        expect(preview).not_to_have_attribute("src", before)
        before = preview.get_attribute("src")
        edit.click()
        template_select = dialog.get_by_role("combobox")
        template_select.click()
        page.get_by_role("option", name="远方邮笺", exact=True).click()
        export.click()
        expect(preview).not_to_have_attribute("src", before)
        preview.evaluate("el => el.decode()")
        expect(
            dialog.get_by_text("远方邮笺  /  1500 × 1000 px · 预览与下载一致", exact=True)
        ).to_be_visible()
        page.screenshot(path=str(output / "dialog_postal.png"), full_page=True)
        preview_response = page.request.get(preview.get_attribute("src"))
        assert preview_response.ok
        with page.expect_download() as download_info:
            dialog.get_by_role("button", name="下载这封来信 PNG", exact=True).click()
        download = download_info.value
        target = output / "browser_download.png"
        download.save_as(str(target))
        assert target.read_bytes() == preview_response.body()
        with Image.open(io.BytesIO(target.read_bytes())) as image:
            assert image.size == (1500, 1000) and image.format == "PNG"
        report["checks"].append(
            "real image generation, dialog editing, template change, exact PNG download"
        )
        dialog.get_by_role("button", name="完成，返回风景", exact=True).click()
        expect(dialog).not_to_be_visible()
        compose.click()
        edit.click()
        expect(dialog.get_by_label("明信片标题", exact=True)).to_have_value("寄自山海之间")
        page.keyboard.press("Escape")
        expect(dialog).not_to_be_visible()
        # A normal rerun after Escape must not reopen the modal.
        page.get_by_role("button", name="生成双图对照", exact=True).click()
        expect(compose).to_be_enabled()
        expect(dialog).not_to_be_visible()
        report["checks"].append("draft retained; Done and Escape dismiss without reopening")
        pictures = page.locator(".st-key-result_pair img")
        expect(pictures).to_have_count(2, timeout=180000)
        for picture in pictures.all():
            picture.evaluate("el => el.decode()")
            assert picture.evaluate("el => el.naturalWidth") == 512
        page.screenshot(path=str(output / "comparison.png"), full_page=True)
        page.get_by_role("button", name="对照范围").click()
        page.get_by_text("局部 4×", exact=True).click()
        page.keyboard.press("Escape")
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.st-key-result_pair img'))"
            ".every(el => el.naturalWidth === 128)"
        )
        report["checks"].append("equal full-resolution panels and synchronized detail crop")
        for width, height in ((1440, 900), (1366, 768), (1280, 720)):
            page.set_viewport_size({"width": width, "height": height})
            for name in ("本次输入", "区域分析", "评分", "结果对照"):
                page.get_by_role("tab", name=name, exact=True).click()
                check_viewport(page)
        lbox, rbox = (picture.bounding_box() for picture in pictures.all())
        assert abs(lbox["y"] - rbox["y"]) <= 1 and lbox["x"] < rbox["x"]
        assert abs(lbox["width"] - rbox["width"]) <= 1
        page.locator('[data-testid="stMain"]').evaluate("el => el.scrollTop = 0")
        page.screenshot(path=str(output / "laptop.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        compose.click()
        expect(dialog).to_be_visible()
        rect = dialog.bounding_box()
        assert rect["x"] >= 0 and rect["x"] + rect["width"] <= 1281
        assert rect["y"] >= 0 and rect["y"] + rect["height"] <= 721
        export.click()
        expect(dialog.get_by_role("button", name="下载这封来信 PNG")).to_be_in_viewport()
        expect(dialog.get_by_role("button", name="完成，返回风景")).to_be_in_viewport()
        report["checks"].append(
            "desktop views, downloads and postcard dialog fit without scrolling"
        )
        assert not errors, errors
        assert not page.locator('[data-testid="stException"]').count()
        report["javascript_errors"] = errors
        report["status"] = "passed"
        (output / "browser_checks.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=True))
        browser.close()


if __name__ == "__main__":
    main()
