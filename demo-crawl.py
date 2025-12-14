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
# Thiết lập ROOT, để import minio_client
# =========================
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "minio"))
from minio_client import get_minio_client  # type: ignore


class TikTokScraper:
    """
    Lớp chính:
    - Điều khiển Playwright (navigate, scroll, captcha)
    - Thu thập metadata + comments
    - Tải video bằng yt_dlp
    - Upload video + metadata lên MinIO
    - Ghi nhận video_id đã crawl để tránh xử lý trùng
    """

    def __init__(self, config_file: str = "config.yaml") -> None:
        # -------------------------------------------------
        # 1. Load cấu hình từ YAML
        # -------------------------------------------------
        config_path = Path(__file__).parent / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Playwright settings
        self.TIMEOUT: int = int(self.cfg.get("timeout_ms", 10_000))
        self.VIEWPORT: Dict[str, int] = {
            "width": int(self.cfg.get("viewport_width", 1280)),
            "height": int(self.cfg.get("viewport_height", 720)),
        }

        # Crawl behavior
        self.MAX_COMMENT_ROUNDS: int = int(self.cfg.get("max_comments_rounds", 40))
        self.MIN_SLEEP: float = float(self.cfg.get("min_sleep_sec", 1.0))
        self.MAX_SLEEP: float = float(self.cfg.get("max_sleep_sec", 3.0))
        self.DOWNLOAD_VIDEO: bool = bool(self.cfg.get("download_video", True))

        # Giới hạn comment:
        # - Tổng (level1 + level2) tối đa 200
        # - Mỗi comment level1 tối đa 15 reply (level2)
        self.MAX_TOTAL_COMMENTS: int = int(self.cfg.get("max_total_comments", 200))
        # Headless mode: Env var > Config > Default True
        env_headless = os.getenv("HEADLESS")
        if env_headless is not None:
             self.HEADLESS = env_headless.lower() == "true"
        else:
             self.HEADLESS = bool(self.cfg.get("headless", True))


        self.MAX_REPLIES_PER_COMMENT: int = int(
            self.cfg.get("max_replies_per_comment", 15)
        )

        # Hashtags
        self.HASHTAGS: List[str] = list(self.cfg.get("hashtags", []))
        self.MAX_VIDEOS: int = int(self.cfg.get("max_videos_per_hashtag", 10))

        # Folders
        export_dir_cfg = self.cfg.get("export_dir", "exports")
        log_file_cfg = self.cfg.get("log_file", "logs/tiktok_scraper.log")
        tmp_dir_cfg = self.cfg.get("tmp_download_dir", "tmp_downloads")

        self.EXPORT_DIR: Path = ROOT / export_dir_cfg
        self.LOG_FILE: Path = ROOT / log_file_cfg
        self.TMP_DOWNLOAD_DIR: Path = ROOT / tmp_dir_cfg

        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.TMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # File lưu danh sách video_id đã xử lý
        self.DOWNLOADED_IDS_FILE: Path = Path(__file__).parent / "downloaded_videos.txt"

        # -------------------------------------------------
        # 2. Cấu hình logging theo config
        # -------------------------------------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.LOG_FILE, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

        # Fixed User Agent to avoid detection
        self.USER_AGENT: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        # -------------------------------------------------
        # 3. Kết nối MinIO
        # -------------------------------------------------
        minio_cfg_file = self.cfg.get("minio_config", "config_tiktok.yaml")
        # Ensure secure=False for docker environment
        os.environ["MINIO_SECURE"] = "False"
        self.minio_client, self.bucket_name = get_minio_client(minio_cfg_file)


        # -------------------------------------------------
        # 4. Load danh sách video đã xử lý
        # -------------------------------------------------
        self.downloaded_ids: Set[str] = set()
        self._load_downloaded_ids()

        logging.info(f"[CONFIG] Loaded from {config_path}")
        logging.info(f"[MINIO] Using bucket: {self.bucket_name}")
        logging.info(
            f"[CRAWL] Hashtags: {self.HASHTAGS}, max_videos_per_hashtag={self.MAX_VIDEOS}"
        )
        logging.info(
            f"[PATH] EXPORT_DIR={self.EXPORT_DIR}, LOG_FILE={self.LOG_FILE}, "
            f"TMP_DOWNLOAD_DIR={self.TMP_DOWNLOAD_DIR}"
        )
        logging.info(
            f"[STATE] Downloaded IDs file: {self.DOWNLOADED_IDS_FILE} "
            f"({len(self.downloaded_ids)} IDs loaded)"
        )
        logging.info(
            f"[COMMENTS] MAX_TOTAL_COMMENTS={self.MAX_TOTAL_COMMENTS}, "
            f"MAX_REPLIES_PER_COMMENT={self.MAX_REPLIES_PER_COMMENT}, "
            f"MAX_COMMENT_ROUNDS={self.MAX_COMMENT_ROUNDS}"
        )

    # ---------------------------------------------------------------------
    # Quản lý danh sách video_id đã xử lý
    # ---------------------------------------------------------------------
    def _load_downloaded_ids(self) -> None:
        """Đọc file downloaded_videos.txt để biết những video_id đã xử lý trước đó."""
        if not self.DOWNLOADED_IDS_FILE.exists():
            # Tạo file trống để sau này append
            self.DOWNLOADED_IDS_FILE.touch()
            self.downloaded_ids = set()
            return

        ids: Set[str] = set()
        with self.DOWNLOADED_IDS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    ids.add(vid)

        self.downloaded_ids = ids

    def _mark_downloaded(self, video_id: str) -> None:
        """Ghi nhận rằng video_id này đã được xử lý (crawl + upload)."""
        if video_id in self.downloaded_ids:
            return

        self.downloaded_ids.add(video_id)
        # Append ngay một dòng mới
        with self.DOWNLOADED_IDS_FILE.open("a", encoding="utf-8") as f:
            f.write(video_id + "\n")
        logging.info(f"[STATE] Marked as downloaded: {video_id}")

    # ---------------------------------------------------------------------
    # Helper: sleep random
    # ---------------------------------------------------------------------
    async def _random_sleep(self) -> None:
        """Ngủ random vài giây để trông giống người dùng thật hơn."""
        delay = random.uniform(self.MIN_SLEEP, self.MAX_SLEEP)
        logging.info(f"Sleeping for {delay:.2f} seconds...")
        await asyncio.sleep(delay)

    # ---------------------------------------------------------------------
    # CAPTCHA handler
    # ---------------------------------------------------------------------
    async def _handle_captcha(self, page: Page) -> None:
        """
        Nếu TikTok hiện dialog CAPTCHA / verification thì chờ user solve.
        """
        try:
            captcha_dialog = page.locator('div[role="dialog"]')
            if await captcha_dialog.count() > 0 and await captcha_dialog.is_visible():
                logging.warning("CAPTCHA detected. Solve it manually in the browser...")
                await page.wait_for_selector(
                    'div[role="dialog"]',
                    state="detached",
                    timeout=self.TIMEOUT * 2000,
                )
                logging.info("CAPTCHA solved. Continue scraping...")
                await self._random_sleep()
        except Exception as e:
            logging.error(f"Error handling CAPTCHA: {e}")

    # ---------------------------------------------------------------------
    # Comment panel & replies
    # ---------------------------------------------------------------------
    async def _open_comments_panel(self, page: Page) -> None:
        """
        Mở panel comment (cột bên phải) nếu chưa mở.
        """
        try:
            comment_list = page.locator('[data-e2e="comment-list"]')
            if await comment_list.count() > 0:
                logging.info("Comment panel already visible.")
                return

            comment_button = page.locator('[data-e2e="comment-icon"]')
            if await comment_button.count() > 0:
                logging.info("Clicking comment button to open comments panel...")
                await comment_button.first.click()
                await self._random_sleep()
            else:
                logging.warning("Cannot find comment button. Comments may be disabled.")
        except Exception as e:
            logging.error(f"Error in _open_comments_panel: {e}")

    async def _expand_all_replies(self, page: Page) -> None:
        """
        Click các nút 'View replies' trong DivViewRepliesContainer.
        Không giới hạn số lượng reply trong DOM, nhưng sau mỗi vòng scroll
        ta sẽ dừng toàn bộ quá trình khi đã đủ MAX_TOTAL_COMMENTS.
        """
        try:
            containers = page.locator("div[class*='DivViewRepliesContainer']")
            count = await containers.count()
            if count == 0:
                return

            logging.info(f"Found {count} reply containers in this round")

            for i in range(count):
                try:
                    text = (await containers.nth(i).inner_text()).strip().lower()
                except Exception:
                    continue

                if "view" not in text:
                    continue

                try:
                    await containers.nth(i).click()
                    await self._random_sleep()
                except Exception as e:
                    logging.debug(f"Failed to click reply span idx={i}: {e}")

        except Exception as e:
            logging.error(f"Error in _expand_all_replies: {e}")

    async def _collect_comments_tree(self, page: Page) -> List[Dict[str, Any]]:
        """
        Crawl comment TikTok dạng cây:
        - comment-level-1: comment gốc
        - comment-level-2: reply của comment gần nhất

        Giới hạn:
        - Mỗi comment level 1: tối đa self.MAX_REPLIES_PER_COMMENT reply (level 2)
        - Tổng số comment (level1 + level2): tối đa self.MAX_TOTAL_COMMENTS

        Trong quá trình scroll:
        - Mỗi vòng: scroll + expand replies + ngủ
        - Sau mỗi vòng: đếm tổng số node comment đang có trong DOM,
          nếu >= MAX_TOTAL_COMMENTS thì dừng, không lướt nữa.
        """
        await self._open_comments_panel(page)

        # Scroll nhiều lần để load thêm comment + mở reply,
        # nhưng dừng sớm nếu tổng số comment đủ.
        for round_idx in range(self.MAX_COMMENT_ROUNDS):
            logging.info(f"[COMMENTS] Scroll round {round_idx + 1}/{self.MAX_COMMENT_ROUNDS}")
            try:
                await page.mouse.wheel(0, 1500)
            except Exception as e:
                logging.error(f"Error while scrolling comments: {e}")
                break

            await self._expand_all_replies(page)
            await self._random_sleep()

            # Đếm số comment level (1 & 2) đang có trong DOM
            try:
                total_nodes = await page.evaluate(
                    """
                    () => {
                        const nodes = document.querySelectorAll('[data-e2e^="comment-level"]');
                        return nodes.length;
                    }
                    """
                )
            except Exception as e:
                logging.error(f"Error counting comments in DOM: {e}")
                total_nodes = 0

            logging.info(
                f"[COMMENTS] DOM currently has {total_nodes} comment nodes "
                f"(limit={self.MAX_TOTAL_COMMENTS})"
            )

            if isinstance(total_nodes, int) and total_nodes >= self.MAX_TOTAL_COMMENTS:
                logging.info(
                    "[COMMENTS] Reached MAX_TOTAL_COMMENTS in DOM, stop scrolling further."
                )
                break

        # Chờ ít nhất 1 comment top-level
        try:
            await page.wait_for_selector(
                '[data-e2e*="comment-level-1"]', timeout=self.TIMEOUT
            )
        except Exception:
            logging.error("No top-level comments found after scrolling.")
            return []

        # Xây dựng cây comment với giới hạn:
        # - Mỗi root: tối đa MAX_REPLIES_PER_COMMENT replies
        # - Tổng: tối đa MAX_TOTAL_COMMENTS nodes
        try:
            max_total = self.MAX_TOTAL_COMMENTS
            max_replies = self.MAX_REPLIES_PER_COMMENT

            comments_tree = await page.evaluate(
                f"""
                () => {{
                    const MAX_TOTAL = {max_total};
                    const MAX_REPLIES = {max_replies};

                    const nodes = Array.from(
                        document.querySelectorAll('[data-e2e^="comment-level"]')
                    );
                    const tree = [];
                    let current = null;
                    let totalCount = 0;

                    const getTextForNode = (n) => {{
                        const textEl =
                            n.querySelector('[data-e2e="comment-content"]') ||
                            n.querySelector("span");
                        return textEl ? textEl.textContent.trim() : "";
                    }};

                    for (const n of nodes) {{
                        if (totalCount >= MAX_TOTAL) {{
                            break;
                        }}

                        const levelAttr = n.getAttribute("data-e2e") || "";
                        const text = getTextForNode(n);

                        if (!text) continue;

                        if (levelAttr.includes("comment-level-1")) {{
                            // Comment gốc
                            current = {{
                                text,
                                replies: []
                            }};
                            tree.push(current);
                            totalCount += 1;

                            if (totalCount >= MAX_TOTAL) {{
                                break;
                            }}
                        }} else if (levelAttr.includes("comment-level-2")) {{
                            // Reply cho comment gần nhất
                            if (current && current.replies.length < MAX_REPLIES) {{
                                current.replies.push({{ text }});
                                totalCount += 1;

                                if (totalCount >= MAX_TOTAL) {{
                                    break;
                                }}
                            }}
                        }}
                    }}

                    return tree;
                }}
                """
            )

            logging.info(
                f"Comment tree collected. Root comments: {len(comments_tree)} "
                f"(max_total={self.MAX_TOTAL_COMMENTS}, max_replies_per_root={self.MAX_REPLIES_PER_COMMENT})"
            )
            return comments_tree

        except Exception as e:
            logging.error(f"Error building comment tree: {e}")
            return []

    # ---------------------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------------------
    async def _extract_video_info(self, page: Page, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Mở URL video TikTok, lấy metadata + comments.
        """
        logging.info(f"Extracting info from: {video_url}")

        try:
            await page.goto(video_url, wait_until="domcontentloaded")
            await self._random_sleep()
            await self._handle_captcha(page)

            await self._open_comments_panel(page)

            await page.wait_for_selector('[data-e2e="like-count"]', timeout=self.TIMEOUT)

            video_info: Dict[str, Any] = await page.evaluate(
                """
                () => {
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
                }
                """
            )

            try:
                comments = await self._collect_comments_tree(page)
                video_info["comments_tree"] = comments
            except Exception as e:
                logging.error(f"Error collecting comments for {video_url}: {e}")
                video_info["comments_tree"] = []

            logging.info(f"Successfully extracted info for: {video_url}")
            return video_info

        except Exception as e:
            logging.error(f"Failed to extract info from {video_url}: {e}")
            return None

    # ---------------------------------------------------------------------
    # MinIO utilities
    # ---------------------------------------------------------------------
    def _upload_file_to_minio(self, local_path: Path, object_name: str) -> Optional[str]:
        """
        Upload file từ ổ cứng lên MinIO rồi xóa file local.
        Trả về path dạng s3://bucket/object_name nếu thành công.
        """
        try:
            self.minio_client.fput_object(
                self.bucket_name,
                object_name,
                str(local_path),
            )
            logging.info(f"Uploaded to MinIO: {object_name}")
            local_path.unlink(missing_ok=True)
            return f"s3://{self.bucket_name}/{object_name}"
        except Exception as e:
            logging.error(f"MinIO upload error ({object_name}): {e}")
            return None

    def _save_json_to_minio(self, data: Dict[str, Any], object_name: str) -> None:
        """
        Lưu metadata dạng JSON trực tiếp lên MinIO (không cần file tạm).
        """
        try:
            json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            json_stream = io.BytesIO(json_bytes)

            self.minio_client.put_object(
                self.bucket_name,
                object_name,
                data=json_stream,
                length=len(json_bytes),
                content_type="application/json",
            )
            logging.info(f"Metadata saved to MinIO: {object_name}")
        except Exception as e:
            logging.error(f"Error saving JSON to MinIO ({object_name}): {e}")

    # ---------------------------------------------------------------------
    # Video download (yt_dlp)
    # ---------------------------------------------------------------------
    def _download_video(self, video_url: str, video_id: str) -> Optional[str]:
        """
        Tải video bằng yt_dlp vào thư mục tạm,
        sau đó upload lên MinIO rồi xoá file local.

        Đường dẫn MinIO:
          bronze/{video_id}/video.mp4
        """
        out_tmpl = str(self.TMP_DOWNLOAD_DIR / f"{video_id}.%(ext)s")

        ydl_opts = {
            "outtmpl": out_tmpl,
            "format": "best",
            "quiet": True,
            "ignoreerrors": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                if not info:
                    logging.warning(f"yt_dlp failed to download: {video_url}")
                    return None

                local_filename = Path(ydl.prepare_filename(info))

                object_name = f"bronze/{video_id}/video.mp4"
                s3_path = self._upload_file_to_minio(local_filename, object_name)
                return s3_path

        except Exception as e:
            logging.error(f"Error downloading video {video_url}: {e}")
            return None

    # ---------------------------------------------------------------------
    # Scrape 1 video
    # ---------------------------------------------------------------------
    async def scrape_single_video(self, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Crawl 1 video cụ thể (URL trực tiếp).
        Trả về dict metadata; đã upload video + metadata lên MinIO.

        MinIO:
          bronze/{video_id}/video.mp4
          bronze/{video_id}/metadata.json
        """
        # Lấy video_id từ URL sớm để có thể kiểm tra skip
        try:
            video_id = video_url.split("video/")[-1].split("?")[0]
        except Exception:
            video_id = f"vid_{int(time.time())}"

        # Nếu đã xử lý rồi thì bỏ qua luôn
        if video_id in self.downloaded_ids:
            logging.info(f"[SKIP] Video {video_id} already processed, skip single-video crawl.")
            return None

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.HEADLESS)
            context = await browser.new_context(
                viewport=self.VIEWPORT,
                user_agent=self.USER_AGENT,
            )
            page = await context.new_page()

            try:
                video_info = await self._extract_video_info(page, video_url)
                if not video_info:
                    raise RuntimeError("Failed to extract video info")

                if self.DOWNLOAD_VIDEO:
                    s3_path = self._download_video(video_url, video_id)
                    if s3_path:
                        video_info["video_s3_path"] = s3_path

                meta_object = f"bronze/{video_id}/metadata.json"
                self._save_json_to_minio(video_info, meta_object)

                # Đánh dấu đã xử lý
                self._mark_downloaded(video_id)

                return video_info

            except Exception as e:
                logging.error(f"Error scraping single video: {e}")
                return None
            finally:
                await browser.close()

    # ---------------------------------------------------------------------
    # Hashtag: collect URLs + scrape multiple videos
    # ---------------------------------------------------------------------
    async def _collect_video_urls_by_hashtag(
        self, page: Page, hashtag: str, max_videos: int
    ) -> List[str]:
        """
        Mở trang hashtag và lấy danh sách URL video (/video/...).
        """
        hashtag_clean = hashtag.lstrip("#")
        hashtag_url = f"https://www.tiktok.com/tag/{hashtag_clean}"
        logging.info(f"Opening hashtag page: {hashtag_url}")

        await page.goto(hashtag_url, wait_until="domcontentloaded")
        await self._random_sleep()
        
        # --- Pause for Manual CAPTCHA Solving ---
        if not self.HEADLESS:
            print("\n" + "="*50)
            print("Action Required: Please solve the CAPTCHA in the browser window.")
            print("Make sure video thumbnails are visible.")
            input("Press ENTER here after you have solved the CAPTCHA...")
            print("="*50 + "\n")
        
        await self._handle_captcha(page)

        collected_urls: Set[str] = set()
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
            await self._random_sleep()

        return list(collected_urls)

    async def scrape_hashtag(self, hashtag: str, max_videos: int) -> List[Dict[str, Any]]:
        """
        Crawl nhiều video theo hashtag.
        - Tự động lấy list URL video
        - Với mỗi video:
          + Lấy metadata + comment tree
          + (optional) Tải video
          + Lưu metadata + video lên MinIO:

            bronze/{video_id}/video.mp4
            bronze/{video_id}/metadata.json

        - Nếu video_id đã có trong downloaded_videos.txt thì SKIP
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.HEADLESS)
            context = await browser.new_context(
                viewport=self.VIEWPORT,
                user_agent=self.USER_AGENT,
            )
            page = await context.new_page()

            results: List[Dict[str, Any]] = []

            try:
                video_urls = await self._collect_video_urls_by_hashtag(
                    page, hashtag, max_videos=max_videos
                )
                logging.info(f"Found {len(video_urls)} video URLs for hashtag #{hashtag}")

                for idx, video_url in enumerate(video_urls, start=1):
                    # Lấy video_id từ URL sớm để skip nếu đã xử lý
                    try:
                        video_id = video_url.split("video/")[-1].split("?")[0]
                    except Exception:
                        video_id = f"vid_{int(time.time())}"

                    if video_id in self.downloaded_ids:
                        logging.info(
                            f"[SKIP] [{idx}/{len(video_urls)}] video_id={video_id} "
                            f"already processed, skip."
                        )
                        continue

                    logging.info(f"[{idx}/{len(video_urls)}] Scraping video: {video_url}")

                    video_info = await self._extract_video_info(page, video_url)
                    if not video_info:
                        logging.warning(
                            f"Skip {video_url} because metadata extraction failed."
                        )
                        continue

                    if self.DOWNLOAD_VIDEO:
                        s3_path = self._download_video(video_url, video_id)
                        if s3_path:
                            video_info["video_s3_path"] = s3_path

                    meta_object = f"bronze/{video_id}/metadata.json"
                    self._save_json_to_minio(video_info, meta_object)

                    # Đánh dấu đã xử lý
                    self._mark_downloaded(video_id)

                    results.append(video_info)
                    await self._random_sleep()

            except Exception as e:
                logging.error(f"Error scraping hashtag #{hashtag}: {e}")
            finally:
                await browser.close()

            return results

    # ---------------------------------------------------------------------
    # Save kết quả ra file JSON local để debug / demo
    # ---------------------------------------------------------------------
    def save_results_local(self, data: Any, filename: str) -> None:
        out_path = self.EXPORT_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Results saved to {out_path}")


# =====================================================================
# HÀM main() – ví dụ crawl theo hashtag, dùng tham số từ config.yaml
# =====================================================================
async def main() -> None:
    scraper = TikTokScraper()

    if not scraper.HASHTAGS:
        logging.error("No hashtags configured in config.yaml")
        return

    for tag in scraper.HASHTAGS:
        logging.info(f"=== START SCRAPING HASHTAG #{tag} ===")
        video_data_list = await scraper.scrape_hashtag(
            hashtag=tag,
            max_videos=scraper.MAX_VIDEOS,
        )

        if video_data_list:
            out_file = f"tiktok_{tag}_videos.json"
            scraper.save_results_local(video_data_list, filename=out_file)
            logging.info(
                f"Scraping completed for #{tag}. "
                f"Total videos collected: {len(video_data_list)}"
            )
        else:
            logging.error(f"Failed to scrape videos for #{tag}")


if __name__ == "__main__":
    asyncio.run(main())
