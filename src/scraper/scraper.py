"""
Web scraper module for fetching pages with robots.txt compliance.
"""

import time
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass
import json

import requests
from bs4 import BeautifulSoup
import structlog

from ..common.config import config, load_site_config, MinIOClient
from ..common.lineage import lineage_tracker, LineageContext


@dataclass
class ScrapedPage:
    """Container for scraped page data."""
    url: str
    title: str
    content: str
    html: str
    timestamp: datetime
    status_code: int
    response_time: float
    content_length: int


class RobotChecker:
    """Robots.txt compliance checker."""
    
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.logger = structlog.get_logger(__name__)
    
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robots_cache:
            robots_url = urljoin(base_url, "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                rp.read()
                self.robots_cache[base_url] = rp
                self.logger.info("Loaded robots.txt", robots_url=robots_url)
            except Exception as e:
                self.logger.warning("Failed to load robots.txt", robots_url=robots_url, error=str(e))
                # Create permissive robot parser if robots.txt fails
                rp = RobotFileParser()
                rp.set_url(robots_url)
                self.robots_cache[base_url] = rp
        
        return self.robots_cache[base_url].can_fetch(self.user_agent, url)


class WebScraper:
    """Web scraper with rate limiting and robots.txt compliance."""
    
    def __init__(self, site_config_path: str = "sample_config/site.yaml"):
        self.site_config = load_site_config(site_config_path)
        self.config = config
        self.minio_client = MinIOClient(self.config.minio)
        self.robot_checker = RobotChecker(self.config.scraping.user_agent)
        self.session = requests.Session()
        self.logger = structlog.get_logger(__name__)
        
        # Setup session headers
        self.session.headers.update(self.site_config.get("scraping", {}).get("headers", {}))
        self.session.headers["User-Agent"] = self.config.scraping.user_agent
        
        # Initialize tracking
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
        # Ensure MinIO bucket exists
        self.minio_client.ensure_bucket_exists()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent processing."""
        parsed = urlparse(url)
        # Remove fragment and normalize
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ""  # Remove fragment
        ))
        return normalized
    
    def _should_scrape_url(self, url: str) -> bool:
        """Check if URL should be scraped based on configuration."""
        normalized_url = self._normalize_url(url)
        
        # Check if already scraped
        if normalized_url in self.scraped_urls or normalized_url in self.failed_urls:
            return False
        
        # Check URL patterns
        follow_patterns = self.site_config.get("urls", {}).get("follow_patterns", [])
        ignore_patterns = self.site_config.get("urls", {}).get("ignore_patterns", [])
        
        # Check ignore patterns first
        for pattern in ignore_patterns:
            if re.match(pattern.replace("*", ".*"), normalized_url):
                self.logger.debug("URL matches ignore pattern", url=normalized_url, pattern=pattern)
                return False
        
        # Check follow patterns
        if follow_patterns:
            for pattern in follow_patterns:
                if re.match(pattern.replace("*", ".*"), normalized_url):
                    return True
            return False
        
        return True
    
    def _extract_content(self, html: str, url: str) -> Dict[str, str]:
        """Extract content from HTML using configured selectors."""
        soup = BeautifulSoup(html, 'html.parser')
        extraction_config = self.site_config.get("extraction", {})
        
        # Remove unwanted elements
        for selector in extraction_config.get("remove_selectors", []):
            for element in soup.select(selector):
                element.decompose()
        
        # Extract title
        title_selector = extraction_config.get("title_selector", "h1")
        title_element = soup.select_one(title_selector)
        title = title_element.get_text(strip=True) if title_element else ""
        
        # Extract main content
        content_selector = extraction_config.get("content_selector", "body")
        content_element = soup.select_one(content_selector)
        
        if content_element:
            content = content_element.get_text(separator="\n", strip=True)
        else:
            content = soup.get_text(separator="\n", strip=True)
        
        # Clean content if configured
        if extraction_config.get("clean_text", True):
            # Remove extra whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
        
        # Check minimum content length
        min_length = extraction_config.get("min_text_length", 100)
        if len(content) < min_length:
            self.logger.warning("Content too short", url=url, length=len(content), min_length=min_length)
        
        return {
            "title": title,
            "content": content
        }
    
    def _scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page."""
        normalized_url = self._normalize_url(url)
        
        # Check robots.txt
        if not self.robot_checker.can_fetch(normalized_url):
            self.logger.info("URL blocked by robots.txt", url=normalized_url)
            self.failed_urls.add(normalized_url)
            return None
        
        try:
            start_time = time.time()
            response = self.session.get(
                normalized_url,
                timeout=self.config.scraping.timeout_seconds
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            
            # Extract content
            extracted = self._extract_content(response.text, normalized_url)
            
            # Check quality requirements
            quality_config = self.site_config.get("quality", {})
            skip_indicators = quality_config.get("skip_if_contains", [])
            
            for indicator in skip_indicators:
                if indicator.lower() in response.text.lower():
                    self.logger.info("Page contains skip indicator", url=normalized_url, indicator=indicator)
                    self.failed_urls.add(normalized_url)
                    return None
            
            page = ScrapedPage(
                url=normalized_url,
                title=extracted["title"],
                content=extracted["content"],
                html=response.text,
                timestamp=datetime.utcnow(),
                status_code=response.status_code,
                response_time=response_time,
                content_length=len(response.text)
            )
            
            self.scraped_urls.add(normalized_url)
            self.logger.info("Successfully scraped page", 
                           url=normalized_url, 
                           title=page.title[:50],
                           content_length=page.content_length,
                           response_time=response_time)
            
            return page
            
        except requests.RequestException as e:
            self.logger.error("Failed to scrape page", url=normalized_url, error=str(e))
            self.failed_urls.add(normalized_url)
            return None
    
    def _store_page(self, page: ScrapedPage) -> bool:
        """Store scraped page to MinIO."""
        try:
            # Generate file names
            url_hash = hashlib.md5(page.url.encode()).hexdigest()[:8]
            timestamp_str = page.timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Store HTML
            html_key = f"raw/html/{timestamp_str}_{url_hash}.html"
            html_success = self.minio_client.upload_text(html_key, page.html)
            
            # Store extracted text
            text_key = f"raw/text/{timestamp_str}_{url_hash}.txt"
            text_content = f"URL: {page.url}\nTitle: {page.title}\nTimestamp: {page.timestamp}\n\n{page.content}"
            text_success = self.minio_client.upload_text(text_key, text_content)
            
            # Store metadata
            metadata = {
                "url": page.url,
                "title": page.title,
                "timestamp": page.timestamp.isoformat(),
                "status_code": page.status_code,
                "response_time": page.response_time,
                "content_length": page.content_length,
                "html_key": html_key,
                "text_key": text_key
            }
            
            metadata_key = f"raw/metadata/{timestamp_str}_{url_hash}.json"
            metadata_success = self.minio_client.upload_text(metadata_key, json.dumps(metadata, ensure_ascii=False))
            
            return html_success and text_success and metadata_success
            
        except Exception as e:
            self.logger.error("Failed to store page", url=page.url, error=str(e))
            return False
    
    def scrape_urls(self, urls: List[str]) -> Dict[str, int]:
        """Scrape a list of URLs with rate limiting."""
        stats = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
        
        with LineageContext(
            job_name="web_scraper",
            tracker=lineage_tracker,
            source_code_location="src/scraper/scraper.py"
        ) as lineage_ctx:
            
            for i, url in enumerate(urls):
                if len(self.scraped_urls) >= self.config.scraping.max_pages:
                    self.logger.info("Reached maximum page limit", max_pages=self.config.scraping.max_pages)
                    break
                
                if not self._should_scrape_url(url):
                    stats["skipped"] += 1
                    continue
                
                self.logger.info("Scraping page", url=url, progress=f"{i+1}/{len(urls)}")
                
                page = self._scrape_page(url)
                if page:
                    if self._store_page(page):
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1
                
                # Rate limiting
                if i < len(urls) - 1:  # Don't sleep after last URL
                    time.sleep(self.config.scraping.delay_seconds)
            
            # Add output dataset to lineage
            output_dataset = lineage_tracker.create_output_dataset(
                name="raw_scraped_data",
                namespace="minio://datalake/raw",
                stats={
                    "pages_scraped": stats["successful"],
                    "total_attempts": stats["total_urls"],
                    "success_rate": stats["successful"] / max(stats["total_urls"], 1)
                }
            )
            lineage_ctx.add_output(output_dataset)
        
        self.logger.info("Scraping completed", **stats)
        return stats
    
    def run(self) -> Dict[str, int]:
        """Run the scraper with configured start URLs."""
        start_urls = self.site_config.get("urls", {}).get("start_urls", [])
        if not start_urls:
            raise ValueError("No start URLs configured in site configuration")
        
        self.logger.info("Starting web scraper", 
                        start_urls_count=len(start_urls),
                        max_pages=self.config.scraping.max_pages)
        
        return self.scrape_urls(start_urls)


def main():
    """Main entry point for running the scraper."""
    from ..common.config import setup_logging
    
    setup_logging(config.log_level, config.log_format)
    
    scraper = WebScraper()
    stats = scraper.run()
    
    print(f"Scraping completed: {stats}")


if __name__ == "__main__":
    main()