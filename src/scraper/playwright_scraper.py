"""
Playwright-based scraper for JavaScript-heavy websites.
"""

import asyncio
from typing import Optional, Dict, List
from urllib.parse import urljoin

import structlog
from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup

from .scraper import WebScraper, ScrapedPage, RobotChecker
from ..common.config import config


class PlaywrightScraper(WebScraper):
    """Enhanced scraper with JavaScript rendering using Playwright."""
    
    def __init__(self, site_config_path: str = "sample_config/site.yaml"):
        super().__init__(site_config_path)
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.logger = structlog.get_logger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()
    
    async def start_browser(self):
        """Start Playwright browser."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.config.scraping.playwright_headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            self.logger.info("Started Playwright browser", headless=self.config.scraping.playwright_headless)
        except Exception as e:
            self.logger.error("Failed to start browser", error=str(e))
            raise
    
    async def close_browser(self):
        """Close Playwright browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.logger.info("Closed Playwright browser")
    
    async def _scrape_page_async(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page with Playwright."""
        normalized_url = self._normalize_url(url)
        
        # Check robots.txt
        if not self.robot_checker.can_fetch(normalized_url):
            self.logger.info("URL blocked by robots.txt", url=normalized_url)
            self.failed_urls.add(normalized_url)
            return None
        
        page = None
        try:
            # Create new page
            page = await self.browser.new_page(
                user_agent=self.config.scraping.user_agent
            )
            
            # Navigate to URL
            start_time = asyncio.get_event_loop().time()
            response = await page.goto(
                normalized_url,
                wait_until='networkidle',
                timeout=self.config.scraping.playwright_timeout
            )
            response_time = asyncio.get_event_loop().time() - start_time
            
            if not response or response.status >= 400:
                self.logger.error("Failed to load page", url=normalized_url, status=response.status if response else None)
                self.failed_urls.add(normalized_url)
                return None
            
            # Wait for content to load
            await page.wait_for_load_state('domcontentloaded')
            
            # Get page content after JavaScript execution
            html_content = await page.content()
            
            # Extract content using BeautifulSoup
            extracted = self._extract_content(html_content, normalized_url)
            
            # Check quality requirements
            quality_config = self.site_config.get("quality", {})
            skip_indicators = quality_config.get("skip_if_contains", [])
            
            for indicator in skip_indicators:
                if indicator.lower() in html_content.lower():
                    self.logger.info("Page contains skip indicator", url=normalized_url, indicator=indicator)
                    self.failed_urls.add(normalized_url)
                    return None
            
            # Get title from page
            title = await page.title() or extracted["title"]
            
            scraped_page = ScrapedPage(
                url=normalized_url,
                title=title,
                content=extracted["content"],
                html=html_content,
                timestamp=self._get_timestamp(),
                status_code=response.status,
                response_time=response_time,
                content_length=len(html_content)
            )
            
            self.scraped_urls.add(normalized_url)
            self.logger.info("Successfully scraped page with Playwright", 
                           url=normalized_url, 
                           title=scraped_page.title[:50],
                           content_length=scraped_page.content_length,
                           response_time=response_time)
            
            return scraped_page
            
        except Exception as e:
            self.logger.error("Failed to scrape page with Playwright", url=normalized_url, error=str(e))
            self.failed_urls.add(normalized_url)
            return None
        finally:
            if page:
                await page.close()
    
    def _scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Synchronous wrapper for async scraping."""
        # Use the existing event loop if available, otherwise create new one
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            # Fall back to BeautifulSoup scraper
            self.logger.warning("Already in async context, falling back to BeautifulSoup")
            return super()._scrape_page(url)
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self._scrape_page_async(url))
    
    async def scrape_urls_async(self, urls: List[str]) -> Dict[str, int]:
        """Scrape a list of URLs asynchronously."""
        stats = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
        
        await self.start_browser()
        
        try:
            for i, url in enumerate(urls):
                if len(self.scraped_urls) >= self.config.scraping.max_pages:
                    self.logger.info("Reached maximum page limit", max_pages=self.config.scraping.max_pages)
                    break
                
                if not self._should_scrape_url(url):
                    stats["skipped"] += 1
                    continue
                
                self.logger.info("Scraping page with Playwright", url=url, progress=f"{i+1}/{len(urls)}")
                
                page = await self._scrape_page_async(url)
                if page:
                    if self._store_page(page):
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1
                
                # Rate limiting
                if i < len(urls) - 1:
                    await asyncio.sleep(self.config.scraping.delay_seconds)
        finally:
            await self.close_browser()
        
        self.logger.info("Playwright scraping completed", **stats)
        return stats
    
    def run(self) -> Dict[str, int]:
        """Run the scraper with configured start URLs."""
        start_urls = self.site_config.get("urls", {}).get("start_urls", [])
        if not start_urls:
            raise ValueError("No start URLs configured in site configuration")
        
        self.logger.info("Starting Playwright web scraper", 
                        start_urls_count=len(start_urls),
                        max_pages=self.config.scraping.max_pages)
        
        # Run async scraping
        return asyncio.run(self.scrape_urls_async(start_urls))
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow()


def create_scraper(site_config_path: str = "sample_config/site.yaml") -> WebScraper:
    """Factory function to create appropriate scraper based on configuration."""
    if config.scraping.engine == "playwright":
        return PlaywrightScraper(site_config_path)
    else:
        return WebScraper(site_config_path)


def main():
    """Main entry point for running the scraper."""
    from ..common.config import setup_logging
    
    setup_logging(config.log_level, config.log_format)
    
    scraper = create_scraper()
    stats = scraper.run()
    
    print(f"Scraping completed: {stats}")


if __name__ == "__main__":
    main() 