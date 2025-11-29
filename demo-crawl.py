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
                    timeout=self.TIMEOUT * 2000
                )
                logging.info("CAPTCHA solved. Resuming...")
                await self.random_sleep(0.5, 1)
        except Exception as e:
            logging.error(f"Error handling CAPTCHA: {str(e)}")

    # =========================
    # >>> NEW: HÀM MỞ PANEL COMMENT
    # =========================
    async def open_comments_panel(self, page):
        """
        Đảm bảo panel comment đã mở (desktop thường là cột bên phải).
        """
        try:
            # Thử xem list comment đã xuất hiện chưa
            comment_list = page.locator('[data-e2e="comment-list"]')
            if await comment_list.count() > 0:
                logging.info("Comment panel already visible.")
                return

            # Nếu chưa có, thử bấm nút comment
            comment_button = page.locator('[data-e2e="comment-icon"]')
            if await comment_button.count() > 0:
                logging.info("Clicking comment button to open comments panel...")
                await comment_button.first.click()
                await self.random_sleep(1, 2)
            else:
                logging.warning("Cannot find comment button. Maybe comments are disabled?")
        except Exception as e:
            logging.error(f"Error open_comments_panel: {e}")

    
    async def expand_all_replies(self, page):
        """
        Click tất cả các nút 'View replies' bằng DivViewRepliesContainer cho đến khi không còn nút nào.
        """
        try:
            while True:
                containers = page.locator("div[class*='DivViewRepliesContainer']")
                count = await containers.count()

                if count == 0:
                    break

                logging.info(f"Found {count} reply containers")
                clickable_indices = []

                for i in range(count):
                    try:
                        text = (await containers.nth(i).inner_text()).strip().lower()
                    except Exception:
                        continue

                    # Chỉ click những span còn chữ 'view' để tránh lặp vô tận
                    if "view" in text:
                        clickable_indices.append(i)

                if not clickable_indices:
                    logging.info("No more 'View ... replies' spans found, stop expanding replies.")
                    break

                #logging.info(f"[expand_all_replies] loop {loop}, clicking {len(clickable_indices)} reply toggles.")

                for idx in clickable_indices:
                    try:
                        await containers.nth(idx).click()
                        await self.random_sleep(0.2, 0.5)
                    except Exception as e:
                        logging.debug(f"Failed to click reply span idx={idx}: {e}")

                await self.random_sleep(0.5, 1.0)

        except Exception as e:
            logging.error(f"Error expand_all_replies: {e}")
   

    # =========================
    # >>> NEW: HÀM MỞ RỘNG TOÀN BỘ REPLY CỦA COMMENT

    async def collect_comments_tree(self, page, max_rounds=40):
        """
        Crawl comment TikTok dạng cây cho DOM hiện tại:
        - comment-level-1 = comment gốc
        - comment-level-2 = reply của comment ngay phía trên
        """

        # Mở panel comment
        await self.open_comments_panel(page)

        # Scroll load comment + mở reply nhiều lần
        for _ in range(max_rounds):
            try:
                await page.mouse.wheel(0, 1500)
            except Exception:
                break
            await self.expand_all_replies(page)
            await self.random_sleep(0.5, 1.0)

        # Chờ ít nhất 1 comment gốc
        try:
            await page.wait_for_selector('span[data-e2e*="comment-level-1"]', timeout=15000)
        except Exception:
            logging.error("No comment-level-1 found after scrolling.")
            return []

        # Build tree từ DOM
        try:
            comments_tree = await page.evaluate(
                """
                () => {
                    const nodes = Array.from(
                        document.querySelectorAll('span[data-e2e^="comment-level"]')
                    );
                    const tree = [];
                    let current = null;

                    nodes.forEach(n => {
                        const levelAttr = n.getAttribute("data-e2e") || "";
                        const textEl =
                            n.querySelector('[data-e2e="comment-content"]') ||
                            n.querySelector("span");
                        const text = textEl ? textEl.textContent.trim() : "";

                        if (!text) return;

                        if (levelAttr.includes("comment-level-1")) {
                            // Comment gốc
                            current = { text, replies: [] };
                            tree.push(current);
                        }
                        else if (levelAttr.includes("comment-level-2")) {
                            // Reply của comment gần nhất
                            if (current) {
                                current.replies.push({ text });
                            }
                        }
                    });

                    return tree;
                }
                """
            )

            logging.info(f"Comment tree collected, root comments: {len(comments_tree)}")
            return comments_tree

        except Exception as e:
            logging.error(f"Error building comment tree: {e}")
            return []




    async def extract_video_info(self, page, video_url):
        """Extract video details from a TikTok video page."""
        logging.info(f"Extracting info from: {video_url}")

        try:
            await page.goto(video_url, wait_until="networkidle")
            await self.random_sleep(2, 4)
            await self.handle_captcha(page)

            # Đảm bảo comment panel mở sớm (để TikTok load comment trong lúc mình chờ)
            await self.open_comments_panel(page)   # >>> NEW

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
                const getDescription = () => {
                    const selectors = [
                    'span[data-e2e^="desc-span"]',        
                    '[data-e2e="browse-video-desc"]',     
                    'div[data-e2e="browse-video-desc"] span'
                ];

                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                    return el.textContent.trim();
                    }
                }
                return 'N/A';
                };

                return {
                    likes: getTextContent(['[data-e2e="like-count"]', '[data-e2e="browse-like-count"]']),
                    comments: getTextContent(['[data-e2e="comment-count"]', '[data-e2e="browse-comment-count"]']),
                    shares: getTextContent(['[data-e2e="share-count"]']),
                    bookmarks: getTextContent(['[data-e2e="undefined-count"]']),
                    views: getTextContent(['[data-e2e="video-views"]']),
                    description: getDescription(),
                    musicTitle: getTextContent(['.css-pvx3oa-DivMusicText']),
                    date: getTextContent(['span[data-e2e="browser-nickname"] span:last-child']),
                    author: getTextContent(['a[data-e2e="video-author-uniqueid"]']),
                    tags: getTags(),
                    videoUrl: window.location.href
                };
            }"""
            )

            # >>> NEW: Crawl toàn bộ comment + reply
            try:
                comments = await self.collect_comments_tree(page)
                video_info["comments"] = comments
            except Exception as e:
                logging.error(f"Error collecting comments for {video_url}: {e}")
                video_info["comments"] = []

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
        Returns a list of dicts with info for each video (bao gồm comments nếu crawl được).
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
    hashtag = ["xuhuong"]   # sửa hashtag tùy ý: "golf", "football", "coding", ...
    max_videos = 1      # chỉ crawl tối đa 10 video để thử nghiệm

# Previously a triple-quoted block was used to skip the hashtag loop; convert it to comments
    for tag in hashtag:
     video_data_list = await scraper.scrape_hashtag(
         tag,
         max_videos=max_videos
     )
    if video_data_list:
         out_file = f"tiktok_{tag}_videos.json"
         scraper.save_results(video_data_list, filename=out_file)
         logging.info(
             f"\nScraping completed for #{tag}. "
             f"Total videos collected: {len(video_data_list)}"
         )
    else:
         logging.error(f"Failed to scrape videos for #{tag}")
    # -------------------------
    # NẾU MUỐN TEST 1 VIDEO CỤ THỂ, BẬT ĐOẠN NÀY:
    # -------------------------
    #video_url = "https://vt.tiktok.com/ZSfdoKbN1/"
    #video_data = await scraper.scrape_single_video(video_url)
    #if video_data:
        # scraper.save_results(video_data, filename="tiktok_single_video.json")
    # -------------------------


if __name__ == "__main__":
    asyncio.run(main())
