import asyncio
import io
import json
import logging
import random
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
import yt_dlp
from playwright.async_api import Page, async_playwright

# =========================
# Setup Path (Running from data-ingestion/tiktok_crawl)
# =========================
ROOT = Path(__file__).resolve().parents[2] # Project Root
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "minio"))

from minio_client import get_minio_client  # type: ignore

class TikTokScraper:
    """
    Robust TikTok Scraper ported from demo-crawl.py
    Features:
    - Playwright for infinite scroll & metadata extraction (including Comments!)
    - yt-dlp for video download
    - MinIO upload (Bronze Layer)
    - State tracking (downloaded_videos.txt)
    """

    def __init__(self, config_file: str = "config.yaml") -> None:
        # Load Config
        config_path = Path(__file__).parent / config_file
        if not config_path.exists():
             logging.warning(f"Config not found at {config_path}, using empty defaults.")
             self.cfg = {}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        # Settings
        self.TIMEOUT = int(self.cfg.get("timeout_ms", 10000))
        self.VIEWPORT = {
            "width": int(self.cfg.get("viewport_width", 1280)),
            "height": int(self.cfg.get("viewport_height", 720)),
        }
        self.HASHTAGS = list(self.cfg.get("hashtags", []))
        self.MAX_VIDEOS = int(self.cfg.get("max_videos_per_hashtag", 10))
        self.DOWNLOAD_VIDEO = bool(self.cfg.get("download_video", True))
        self.MAX_COMMENT_ROUNDS = int(self.cfg.get("max_comments_rounds", 3)) # Default 3 rounds for speed in demo
        self.MAX_TOTAL_COMMENTS = int(self.cfg.get("max_total_comments", 20)) # Limit total comments
        
        # Paths
        self.EXPORT_DIR = ROOT / self.cfg.get("export_dir", "exports")
        self.TMP_DIR = ROOT / self.cfg.get("tmp_download_dir", "tmp_downloads")
        self.LOG_FILE = ROOT / self.cfg.get("log_file", "logs/tiktok_scraper.log")
        
        # State File (in current dir)
        self.DOWNLOADED_IDS_FILE = Path(__file__).parent / "downloaded_videos.txt"

        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.LOG_FILE, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("tiktok-scraper")

        # Headless
        env_headless = os.getenv("HEADLESS")
        if env_headless is not None:
             self.HEADLESS = env_headless.lower() == "true"
        else:
             self.HEADLESS = bool(self.cfg.get("headless", True)) 

        # MinIO
        minio_cfg_file = self.cfg.get("minio_config", "config_tiktok.yaml")
        self.minio_client, self.bucket_name = get_minio_client(minio_cfg_file)

        # Load State
        self.downloaded_ids: Set[str] = set()
        self._load_downloaded_ids()

        self.USER_AGENT = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.EXPLORE_MODE = False

    def _load_downloaded_ids(self):
        if not self.DOWNLOADED_IDS_FILE.exists():
            self.DOWNLOADED_IDS_FILE.touch()
            return
        
        with open(self.DOWNLOADED_IDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    self.downloaded_ids.add(vid)
        
        self.logger.info(f"Loaded {len(self.downloaded_ids)} processed IDs.")

    def _mark_downloaded(self, video_id: str):
        if video_id not in self.downloaded_ids:
            self.downloaded_ids.add(video_id)
            with open(self.DOWNLOADED_IDS_FILE, "a", encoding="utf-8") as f:
                f.write(video_id + "\n")
            self.logger.info(f"[STATE] Marked {video_id} as processed.")

    async def _handle_captcha(self, page: Page):
        try:
            # Check for CAPTCHA dialog with a short timeout to let it appear
            # Using a broad selector, but TikTok often puts captcha in a role="dialog" or specific iframe
            # We'll try to wait 5s for it to show up
            captcha_selector = 'div[role="dialog"]' # Generic fallback
            # Note: TikTok captcha often has id="captcha_container" or class="captcha-verify-container" 
            # But role="dialog" is a safe catch-all for modal popups (including login prompts which block too)

            found = False
            try:
                await page.wait_for_selector(captcha_selector, state="visible", timeout=5000)
                found = True
            except:
                pass

            if found:
                self.logger.warning("CAPTCHA / Dialog detected!")
                if not self.HEADLESS:
                     self.logger.info("👉 ACTION REQUIRED: Please solve CAPTCHA in the browser. Waiting up to 120s...")
                     try:
                        # Wait for it to disappear
                        await page.wait_for_selector(captcha_selector, state="detached", timeout=120000)
                        self.logger.info("CAPTCHA/Dialog disappeared. Resuming...")
                        await self._random_sleep(2, 4) # Settle down
                     except Exception as e:
                        self.logger.warning(f"Timed out waiting for CAPTCHA solution: {e}")
                else:
                    self.logger.warning("Running headless - cannot solve CAPTCHA manually.")
        except Exception as e:
            self.logger.debug(f"Captcha check error: {e}")

    async def _random_sleep(self, min_s=1.0, max_s=3.0):
        await asyncio.sleep(random.uniform(min_s, max_s))

    # --- Comment Extraction ---
    async def _open_comments_panel(self, page: Page):
        try:
            comment_list = page.locator('[data-e2e="comment-list"]')
            if await comment_list.count() > 0: return

            comment_button = page.locator('[data-e2e="comment-icon"]')
            if await comment_button.count() > 0:
                await comment_button.first.click()
                await self._random_sleep(1, 2)
        except: pass

    async def _collect_comments_tree(self, page: Page) -> List[Dict]:
        await self._open_comments_panel(page)
        
        # Scroll comments
        for _ in range(self.MAX_COMMENT_ROUNDS):
             try:
                 await page.mouse.wheel(0, 1500)
                 await self._random_sleep(0.5, 1.5)
             except: break
             
             # Stop if enough comments
             try:
                 count_js = "document.querySelectorAll('[data-e2e^=\"comment-level\"]').length"
                 count = await page.evaluate(count_js)
                 if count >= self.MAX_TOTAL_COMMENTS: break
             except: pass

        # Extract
        try:
            return await page.evaluate(f"""() => {{
                const MAX = {self.MAX_TOTAL_COMMENTS};
                const nodes = Array.from(document.querySelectorAll('[data-e2e^="comment-level"]'));
                const tree = [];
                let count = 0;
                
                for(const n of nodes) {{
                    if(count >= MAX) break;
                    const textEl = n.querySelector('[data-e2e="comment-content"]') || n.querySelector("span");
                    const text = textEl ? textEl.textContent.trim() : "";
                    
                    if(text) {{
                        tree.push({{ text: text }}); // Simplify tree to list for now or keep flat
                        count++;
                    }}
                }}
                return tree;
            }}""")
        except Exception as e:
            self.logger.error(f"Comment extract error: {e}")
            return []

    # --- Crawl Logic ---

    async def _extract_video_metadata(self, page: Page, url: str) -> Optional[Dict]:
        self.logger.info(f"Extracting metadata: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await self._random_sleep(1, 2)
            await self._handle_captcha(page)
            
            # Simple extraction via JS
            data = await page.evaluate("""() => {
                const getText = (sel) => document.querySelector(sel)?.textContent.trim() || '';
                return {
                    description: getText('[data-e2e="browse-video-desc"]'),
                    likes: getText('[data-e2e="like-count"]'),
                    // comments count, not content
                    comments_count: getText('[data-e2e="comment-count"]'), 
                    shares: getText('[data-e2e="share-count"]'),
                    author: getText('[data-e2e="video-author-uniqueid"]'),
                    date: getText('[data-e2e="browser-nickname"] span:last-child'),
                    tags: Array.from(document.querySelectorAll('a[href*="/tag/"]')).map(e => e.textContent.trim())
                }
            }""")
            
            # Now Crawl Comments Content
            data['comments_tree'] = await self._collect_comments_tree(page)

            data['url'] = url
            data['timestamp'] = time.time()
            data['source'] = 'tiktok_crawler'
            
            try:
                vid = url.split("/video/")[-1].split("?")[0]
            except:
                vid = f"vid_{int(time.time())}"
            data['video_id'] = vid

            return data

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return None

    def _download_and_upload(self, url: str, video_id: str) -> bool:
        """Downloads video using yt-dlp and moves to MinIO Bronze."""
        try:
            temp_path = self.TMP_DIR / f"{video_id}.mp4"
            if temp_path.exists(): temp_path.unlink() # Cleanup old

            os.system(f'touch cookies-tiktok.txt') # Ensure exists

            ydl_opts = {
                'outtmpl': str(temp_path),
                'format': 'best',
                'quiet': True,
                'no_warnings': True,
                'cookiefile': 'cookies-tiktok.txt' # Important for authenticating
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if temp_path.exists():
                minio_path = f"bronze/{video_id}/video.mp4"
                self.minio_client.fput_object(self.bucket_name, minio_path, str(temp_path))
                self.logger.info(f"Deployed to MinIO: s3://{self.bucket_name}/{minio_path}")
                temp_path.unlink()
                return True
            else:
                self.logger.error(f"Download failed: File not found {temp_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Download error for {video_id}: {e}")
            return False

    def _save_meta_minio(self, data: Dict, video_id: str):
        """Saves metadata json to MinIO Bronze."""
        try:
             json_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
             json_stream = io.BytesIO(json_bytes)
             minio_path = f"bronze/{video_id}/metadata.json"
             self.minio_client.put_object(
                 self.bucket_name, minio_path, json_stream, len(json_bytes), content_type="application/json"
             )
        except Exception as e:
             self.logger.error(f"MinIO Meta Upload Error: {e}")

    async def _generate_urls(self, page: Page, start_url: str, max_videos: int):
        """Yields video URLs as they are found, handling scrolling."""
        self.logger.info(f"Navigating to {start_url}")
        
        try:
            await page.goto(start_url, wait_until="domcontentloaded")
        except Exception as e:
            self.logger.error(f"Goto failed: {e}")
            return

        await self._random_sleep(2, 4)
        await self._handle_captcha(page)

        seen_urls = set() # Track seen in this session to avoid yielding duplicates
        last_height = 0
        retries = 0
        total_yielded = 0
        
        while True:
            # check limit
            if max_videos > 0 and total_yielded >= max_videos:
                self.logger.info(f"Reached limit of {max_videos} videos.")
                break

            # Scrape current view
            anchors = await page.query_selector_all('a[href*="/video/"]')
            new_urls = []
            for a in anchors:
                href = await a.get_attribute("href")
                if href and "/video/" in href:
                    full_url = href if href.startswith("http") else f"https://www.tiktok.com{href}"
                    clean_url = full_url.split("?")[0]
                    if clean_url not in seen_urls:
                        seen_urls.add(clean_url)
                        new_urls.append(clean_url)
            
            # Yield new ones
            for url in new_urls:
                yield url
                total_yielded += 1
                if max_videos > 0 and total_yielded >= max_videos:
                     break
                     
            if max_videos > 0 and total_yielded >= max_videos:
                break

            # Scroll
            try:
                new_height = await page.evaluate("document.body.scrollHeight")
            except:
                break

            if new_height == last_height:
                retries += 1
                if retries > 10: # More tolerance/retries for infinite
                    self.logger.warning("No new content found after 10 scrolls. Stopping.")
                    break
                await self._random_sleep(1, 2)
            else:
                retries = 0
                last_height = new_height
            
            await page.mouse.wheel(0, 1500)
            await self._random_sleep(1.5, 3) # Slightly slower to act human

    async def run_crawl(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.HEADLESS)
            # Page 1: Infinite Scroll
            page = await browser.new_page(viewport=self.VIEWPORT, user_agent=self.USER_AGENT)
            # Page 2: Detail Extraction
            page_detail = await browser.new_page(viewport=self.VIEWPORT, user_agent=self.USER_AGENT)
            
            sources = []
            if self.EXPLORE_MODE:
                sources.append(("Explore Feed", "https://www.tiktok.com/explore"))
            elif self.HASHTAGS:
                for tag in self.HASHTAGS:
                    sources.append((f"#{tag}", f"https://www.tiktok.com/tag/{tag}"))
            else:
                self.logger.warning("No hashtags or explore mode selected.")
            
            for source_name, start_url in sources:
                self.logger.info(f"=== Processing {source_name} (Limit: {'Infinite' if self.MAX_VIDEOS <= 0 else self.MAX_VIDEOS}) ===")
                
                async for url in self._generate_urls(page, start_url, self.MAX_VIDEOS):
                    try:
                        vid = url.split("/video/")[-1].split("?")[0]
                    except:
                        vid = f"vid_{int(time.time())}"
                        
                    if vid in self.downloaded_ids:
                        continue
                    
                    self.logger.info(f"Processing {vid}...")
                    
                    try:
                        # Ensure page_detail is alive
                        if page_detail.is_closed():
                            self.logger.warning("⚠️ Detail page was closed! Attempting to recreate...")
                            try:
                                page_detail = await browser.new_page(viewport=self.VIEWPORT, user_agent=self.USER_AGENT)
                            except Exception as e:
                                self.logger.error(f"❌ Could not recreate detail page (Browser likely closed): {e}")
                                break # Stop crawling if browser is dead

                        # Use page_detail to visit video without disturbing scroll state on page
                        meta = await self._extract_video_metadata(page_detail, url)
                        if not meta: continue
                        
                        if self.DOWNLOAD_VIDEO:
                            success = self._download_and_upload(url, vid)
                            if not success: 
                                self.logger.warning(f"Download failed for {vid}, skipping meta upload.")
                                continue
                        
                        self._save_meta_minio(meta, vid)
                        self._mark_downloaded(vid)
                        self.logger.info(f"✅ Crawled {vid}")
                        
                    except Exception as e:
                         self.logger.error(f"Error processing {vid}: {e}")
                    
            await browser.close()
        self.logger.info("Crawl session finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Override max videos per hashtag")
    parser.add_argument("--hashtags", type=str, help="Comma-separated list of hashtags to crawl")
    parser.add_argument("--explore", action="store_true", help="Crawl Explore feed instead of hashtags")
    args = parser.parse_args()
    
    scraper = TikTokScraper()
    
    if args.limit is not None:
        scraper.MAX_VIDEOS = args.limit
        
    if args.explore:
        scraper.EXPLORE_MODE = True
        logging.info("🚀 Explore Mode ACTIVATED! Ignoring hashtags.")
    elif args.hashtags:
        scraper.HASHTAGS = [tag.strip() for tag in args.hashtags.split(",")]
        logging.info(f"Overriding hashtags from CLI: {scraper.HASHTAGS}")
        
    asyncio.run(scraper.run_crawl())
