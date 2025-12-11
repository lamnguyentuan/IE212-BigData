
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Setup Root
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(str(ROOT))

load_dotenv()

# Imports
# Use importlib to import demo-crawl.py as module
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("demo_crawl", str(ROOT / "demo-crawl.py"))
    demo_crawl = importlib.util.module_from_spec(spec)
    sys.modules["demo_crawl"] = demo_crawl
    spec.loader.exec_module(demo_crawl)
    TikTokScraper = demo_crawl.TikTokScraper
except Exception as e:
    print(f"Error importing demo-crawl: {e}")
    sys.exit(1)

async def run_crawl(hashtag):
    print(f"=== Starting Crawl for #{hashtag} ===")
    scraper = TikTokScraper()
    # Scrape 5 videos
    results = await scraper.scrape_hashtag(hashtag, max_videos=5)
    print(f"Crawled {len(results)} videos.")
    for res in results:
         print(f"Saved: {res.get('desc', 'No Desc')[:20]}...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = "vietnam"
        
    asyncio.run(run_crawl(tag))
