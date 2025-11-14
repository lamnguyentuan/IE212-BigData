try:
    from playwright.async_api import async_playwright
except Exception as e:
    raise ImportError(
        "Playwright could not be imported. Install it with:\n"
        "  pip install playwright\n"
        "and then run:\n"
        "  playwright install\n"
        "If you're using a virtual environment, ensure it's activated."
    ) from e

import asyncio
import random
import json
import logging
import time
import os
from urllib.parse import urlparse

import yt_dlp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tiktok_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class TikTokScraper:
    def __init__(self):
        # Để test nhanh: chỉ crawl metadata, KHÔNG tải video
        # Khi muốn tải mp4 thì đổi lại True
        self.DOWNLOAD_VIDEO = True

        self.SAVE_DIR = "downloaded_videos"
        self.USER_AGENT = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.VIEWPORT = {'width': 1280, 'height': 720}
        self.TIMEOUT = 300  # seconds (5 minutes)

    async def random_sleep(self, min_seconds=1, max_seconds=3):
        """Random delay to look more 'human'."""
        delay = random.uniform(min_seconds, max_seconds)
        logging.info(f"Sleeping for {delay:.2f} seconds...")
        await asyncio.sleep(delay)

    async def handle_captcha(self, page):
        """Handling verification codes / CAPTCHA."""
        try:
            captcha_dialog = page.locator('div[role="dialog"]')
            if await captcha_dialog.count() > 0 and await captcha_dialog.is_visible():
                logging.warning("CAPTCHA detected. Please solve it manually.")
                await page.wait_for_selector(
                    'div[role="dialog"]',
                    state='detached',
                    timeout=self.TIMEOUT * 1000
                )
                logging.info("CAPTCHA solved. Resuming...")
                await self.random_sleep(0.5, 1)
        except Exception as e:
            logging.error(f"Error handling CAPTCHA: {str(e)}")

    async def extract_video_info(self, page, video_url):
        """Extract video details from a TikTok video page."""
        logging.info(f"Extracting info from: {video_url}")

        try:
            await page.goto(video_url, wait_until="networkidle")
            await self.random_sleep(2, 4)
            await self.handle_captcha(page)

            # Waiting for key elements to load
            await page.wait_for_selector('[data-e2e="like-count"]', timeout=10000)

            video_info = await page.evaluate(
                """() => {
                const getTextContent = (selectors) => {
                    for (let selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return element.textContent.trim();
                        }
                    }
                    return 'N/A';
                };

                const getTags = () => {
                    const tagElements = document.querySelectorAll('a[data-e2e="search-common-link"]');
                    return Array.from(tagElements).map(el => el.textContent.trim());
                };

                return {
                    likes: getTextContent(['[data-e2e="like-count"]', '[data-e2e="browse-like-count"]']),
                    comments: getTextContent(['[data-e2e="comment-count"]', '[data-e2e="browse-comment-count"]']),
                    shares: getTextContent(['[data-e2e="share-count"]']),
                    bookmarks: getTextContent(['[data-e2e="undefined-count"]']),
                    views: getTextContent(['[data-e2e="video-views"]']),
                    description: getTextContent(['span[data-e2e="new-desc-span"]']),
                    musicTitle: getTextContent(['.css-pvx3oa-DivMusicText']),
                    date: getTextContent(['span[data-e2e="browser-nickname"] span:last-child']),
                    author: getTextContent(['a[data-e2e="video-author-uniqueid"]']),
                    tags: getTags(),
                    videoUrl: window.location.href
                };
            }"""
            )

            logging.info(f"Successfully extracted info for: {video_url}")
            return video_info

        except Exception as e:
            logging.error(f"Failed to extract info from {video_url}: {str(e)}")
            return None

    def download_video(self, video_url):
        """Download video using yt_dlp."""
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        ydl_opts = {
            'outtmpl': os.path.join(self.SAVE_DIR, '%(id)s.%(ext)s'),
            'format': 'best',
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if not info:
                    logging.error("yt_dlp did not return info for this video.")
                    return None
                filename = ydl.prepare_filename(info)
                logging.info(f"Video successfully downloaded: {filename}")
                return filename
        except Exception as e:
            logging.error(f"Error downloading video: {str(e)}")
            return None

    async def scrape_single_video(self, video_url):
        """Scrape a single TikTok video by its direct URL."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport=self.VIEWPORT,
                user_agent=self.USER_AGENT,
            )

            page = await context.new_page()
            result = {}

            try:
                video_info = await self.extract_video_info(page, video_url)
                if not video_info:
                    raise Exception("Failed to extract video info")

                result.update(video_info)

                if self.DOWNLOAD_VIDEO:
                    filename = self.download_video(video_url)
                    if filename:
                        result['local_path'] = filename

            except Exception as e:
                logging.error(f"Error scraping video: {str(e)}")
            finally:
                await browser.close()

            return result

    async def collect_video_urls_by_hashtag(self, page, hashtag, max_videos=20):
        """
        Open hashtag page and collect video URLs (/video/...)
        """
        hashtag = hashtag.lstrip("#")
        hashtag_url = f"https://www.tiktok.com/tag/{hashtag}"
        logging.info(f"Opening hashtag page: {hashtag_url}")

        await page.goto(hashtag_url, wait_until="networkidle")
        await self.random_sleep(2, 4)
        await self.handle_captcha(page)

        collected_urls = set()
        last_height = 0

        while len(collected_urls) < max_videos:
            anchors = await page.query_selector_all('a[href*="/video/"]')
            for a in anchors:
                href = await a.get_attribute("href")
                if not href:
                    continue

                if href.startswith("/"):
                    href = "https://www.tiktok.com" + href

                if "/video/" in href:
                    href = href.split("?")[0]
                    collected_urls.add(href)
                    if len(collected_urls) >= max_videos:
                        break

            logging.info(f"Currently collected {len(collected_urls)} video URLs")

            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                logging.info("No more content to scroll. Stopping.")
                break
            last_height = new_height

            await page.mouse.wheel(0, 2000)
            await self.random_sleep(1, 2)

        return list(collected_urls)

    async def scrape_hashtag(self, hashtag, max_videos=20):
        """
        Crawl multiple TikTok videos by hashtag, no need to know URLs beforehand.
        Returns a list of dicts with info for each video.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport=self.VIEWPORT,
                user_agent=self.USER_AGENT,
            )
            page = await context.new_page()

            results = []

            try:
                # 1. Collect video URLs
                video_urls = await self.collect_video_urls_by_hashtag(
                    page,
                    hashtag,
                    max_videos=max_videos
                )
                logging.info(
                    f"Found {len(video_urls)} video URLs for hashtag #{hashtag}"
                )

                # 2. Scrape each video
                for idx, video_url in enumerate(video_urls, start=1):
                    logging.info(f"[{idx}/{len(video_urls)}] Scraping video: {video_url}")
                    video_info = await self.extract_video_info(page, video_url)
                    if not video_info:
                        logging.warning(
                            f"Skip {video_url} because info extraction failed"
                        )
                        continue

                    if self.DOWNLOAD_VIDEO:
                        filename = self.download_video(video_url)
                        if filename:
                            video_info["local_path"] = filename

                    results.append(video_info)
                    await self.random_sleep(1, 3)

            except Exception as e:
                logging.error(f"Error scraping hashtag #{hashtag}: {str(e)}")
            finally:
                await browser.close()

            return results

    def save_results(self, data, filename="tiktok_video_data.json"):
        """Save the results (single dict or list of dict) to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {filename}")


async def main():
    scraper = TikTokScraper()

    # -------------------------
    # TEST CRAWL 10 VIDEO THEO HASHTAG
    # -------------------------
    hashtag = "xuhuong"   # sửa hashtag tùy ý: "golf", "football", "coding", ...
    max_videos = 10   # chỉ crawl tối đa 10 video để thử nghiệm

    video_data_list = await scraper.scrape_hashtag(
        hashtag,
        max_videos=max_videos
    )

    if video_data_list:
        out_file = f"tiktok_{hashtag}_videos.json"
        scraper.save_results(video_data_list, filename=out_file)
        logging.info(
            f"\nScraping completed for #{hashtag}. "
            f"Total videos collected: {len(video_data_list)}"
        )
    else:
        logging.error(f"Failed to scrape videos for #{hashtag}")

    # -------------------------
    # NẾU MUỐN TEST 1 VIDEO CỤ THỂ, BẬT ĐOẠN NÀY:
    # -------------------------
    # video_url = "https://www.tiktok.com/@petervufriends/video/7476546872253893934"
    # video_data = await scraper.scrape_single_video(video_url)
    # if video_data:
    #     scraper.save_results(video_data, filename="tiktok_single_video.json")
    # -------------------------


if __name__ == "__main__":
    asyncio.run(main())
