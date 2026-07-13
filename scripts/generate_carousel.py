import os
import asyncio
from pathlib import Path

async def main():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[ERROR] Playwright not installed. Please run: pip install playwright && playwright install chromium")
        return

    # Define paths
    base_dir = Path(__file__).parent.parent.resolve()
    html_file = base_dir / "outputs" / "carousel" / "carousel_template.html"
    output_dir = base_dir / "outputs" / "carousel"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not html_file.exists():
        print(f"[ERROR] HTML Template not found at: {html_file}")
        return

    print("Starting Playwright browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Create context with 1080x1350 viewport (LinkedIn Portrait)
        context = await browser.new_context(
            viewport={"width": 1080, "height": 1350},
            device_scale_factor=2 # Render at 2x scale factor for ultra-sharp high-DPI text (2160x2700)
        )
        page = await context.new_page()

        # Load local HTML file
        url = html_file.as_uri()
        print(f"Loading template: {url}")
        await page.goto(url)

        # Wait for fonts and assets to load fully
        print("Waiting for assets to load (2 seconds)...")
        await page.wait_for_timeout(2000)

        # Find all slide elements
        slides = await page.query_selector_all(".slide")
        print(f"Found {len(slides)} slides to render.")

        for idx, slide in enumerate(slides, start=1):
            output_path = output_dir / f"slide_{idx}.png"
            print(f"Rendering slide {idx}/{len(slides)} -> {output_path.name}...")
            
            # Capture the screenshot of the slide element
            await slide.screenshot(
                path=str(output_path),
                type="png",
                omit_background=False
            )

        await browser.close()
        print("\n=== Carousel generation complete! Slides saved to outputs/carousel/ ===")

if __name__ == "__main__":
    asyncio.run(main())
