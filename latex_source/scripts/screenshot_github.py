"""Screenshot a GitHub directory listing for the report's appendix.

Usage:
    python scripts/screenshot_github.py <github-url> <output.png>

Headless Chromium via Playwright, so it needs no browser install of your own:
    pip install playwright && python -m playwright install chromium

A private repository will render as a 404 page here, because the headless browser
is not signed in. Either make the repository public for the screenshot or take it
yourself from a signed-in browser.
"""
import sys

from playwright.sync_api import sync_playwright


def main() -> None:
    url, out = sys.argv[1], sys.argv[2]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000},
                                device_scale_factor=2)   # 2x so text stays sharp in print
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.screenshot(path=out, full_page=False)
        print("title:", page.title())
        browser.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
