"""
Unit tests for the web scraper module.
Tests are written to match the current implementation in src/scraper/scraper.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests

# Try to import scraper components
try:
    from src.scraper.scraper import WebScraper, RobotChecker, ScrapedPage
    from src.common.config import config
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# Skip all tests if scraper is not available
pytestmark = pytest.mark.skipif(not SCRAPER_AVAILABLE, reason="Scraper module not available")


class TestScrapedPage:
    """Test ScrapedPage dataclass."""
    
    def test_scraped_page_creation(self):
        """Test ScrapedPage creation and attributes."""
        timestamp = datetime.utcnow()
        page = ScrapedPage(
            url="https://example.com/test",
            title="Test Page",
            content="This is test content.",
            html="<html><body>This is test content.</body></html>",
            timestamp=timestamp,
            status_code=200,
            response_time=0.5,
            content_length=100
        )
        
        assert page.url == "https://example.com/test"
        assert page.title == "Test Page"
        assert page.content == "This is test content."
        assert page.status_code == 200
        assert page.response_time == 0.5
        assert page.content_length == 100
        assert page.timestamp == timestamp


class TestRobotChecker:
    """Test RobotChecker functionality."""
    
    def test_robot_checker_init(self):
        """Test RobotChecker initialization."""
        user_agent = "TestBot/1.0"
        checker = RobotChecker(user_agent)
        
        assert checker.user_agent == user_agent
        assert checker.robots_cache == {}
    
    @patch('src.scraper.scraper.RobotFileParser')
    def test_can_fetch_allowed(self, mock_robot_parser):
        """Test can_fetch returns True for allowed URLs."""
        # Setup mock
        mock_rp = Mock()
        mock_rp.can_fetch.return_value = True
        mock_robot_parser.return_value = mock_rp
        
        checker = RobotChecker("TestBot/1.0")
        result = checker.can_fetch("https://example.com/page")
        
        assert result is True
        mock_rp.set_url.assert_called_once()
        mock_rp.read.assert_called_once()
        mock_rp.can_fetch.assert_called_once_with("TestBot/1.0", "https://example.com/page")
    
    @patch('src.scraper.scraper.RobotFileParser')
    def test_can_fetch_disallowed(self, mock_robot_parser):
        """Test can_fetch returns False for disallowed URLs."""
        # Setup mock
        mock_rp = Mock()
        mock_rp.can_fetch.return_value = False
        mock_robot_parser.return_value = mock_rp
        
        checker = RobotChecker("TestBot/1.0")
        result = checker.can_fetch("https://example.com/private")
        
        assert result is False
    
    @patch('src.scraper.scraper.RobotFileParser')
    def test_can_fetch_caches_robots(self, mock_robot_parser):
        """Test that robots.txt is cached per domain."""
        mock_rp = Mock()
        mock_rp.can_fetch.return_value = True
        mock_robot_parser.return_value = mock_rp
        
        checker = RobotChecker("TestBot/1.0")
        
        # First call
        checker.can_fetch("https://example.com/page1")
        # Second call to same domain
        checker.can_fetch("https://example.com/page2")
        
        # Should only create one RobotFileParser instance
        assert mock_robot_parser.call_count == 1
        assert mock_rp.read.call_count == 1
    
    @patch('src.scraper.scraper.RobotFileParser')
    def test_can_fetch_robots_error(self, mock_robot_parser):
        """Test can_fetch handles robots.txt loading errors."""
        mock_rp = Mock()
        mock_rp.read.side_effect = Exception("Network error")
        mock_rp.can_fetch.return_value = True  # Should still work with permissive fallback
        mock_robot_parser.return_value = mock_rp
        
        checker = RobotChecker("TestBot/1.0")
        result = checker.can_fetch("https://example.com/page")
        
        # Should handle error gracefully
        assert isinstance(result, bool)


class TestWebScraper:
    """Test WebScraper functionality."""
    
    @patch('src.scraper.scraper.load_site_config')
    @patch('src.scraper.scraper.MinIOClient')
    def test_scraper_init(self, mock_minio_client, mock_load_config):
        """Test WebScraper initialization."""
        mock_config = {
            'scraping': {'headers': {'Accept': 'text/html'}},
            'urls': {'start_urls': ['https://example.com']}
        }
        mock_load_config.return_value = mock_config
        
        # Mock MinIO client
        mock_minio_instance = Mock()
        mock_minio_instance.ensure_bucket_exists.return_value = True
        mock_minio_client.return_value = mock_minio_instance
        
        scraper = WebScraper()
        
        assert scraper.site_config == mock_config
        mock_minio_client.assert_called_once()
        mock_load_config.assert_called_once()
        mock_minio_instance.ensure_bucket_exists.assert_called_once()
    
    def test_normalize_url(self):
        """Test URL normalization."""
        with patch('src.scraper.scraper.load_site_config'), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            
            # Test fragment removal
            result = scraper._normalize_url("https://Example.com/page#section")
            assert result == "https://example.com/page"
            
            # Test case normalization for domain
            result = scraper._normalize_url("HTTPS://EXAMPLE.COM/PAGE")
            assert result == "https://example.com/PAGE"
    
    def test_should_scrape_url_ignore_patterns(self):
        """Test URL filtering with ignore patterns."""
        mock_config = {
            'urls': {
                'ignore_patterns': ['*.pdf', '*/private/*'],
                'follow_patterns': []
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            
            # Should ignore PDF files
            assert not scraper._should_scrape_url("https://example.com/doc.pdf")
            
            # Should ignore private paths
            assert not scraper._should_scrape_url("https://example.com/private/page")
            
            # Should allow other URLs
            assert scraper._should_scrape_url("https://example.com/public/page")
    
    def test_should_scrape_url_follow_patterns(self):
        """Test URL filtering with follow patterns."""
        mock_config = {
            'urls': {
                'ignore_patterns': [],
                'follow_patterns': ['https://example.com/docs/*']
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            
            # Should follow docs URLs
            assert scraper._should_scrape_url("https://example.com/docs/page")
            
            # Should not follow other URLs
            assert not scraper._should_scrape_url("https://example.com/blog/page")
    
    @patch('src.scraper.scraper.config')
    def test_should_scrape_url_already_scraped(self, mock_config):
        """Test URL filtering for already scraped URLs."""
        # Mock the config object properly
        mock_config.scraping.user_agent = "TestBot/1.0"
        mock_config.minio = Mock()
        
        mock_site_config = {
            'scraping': {'headers': {}},
            'urls': {
                'ignore_patterns': [],
                'follow_patterns': []
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_site_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            url = "https://example.com/page"
            
            # First time should be allowed
            assert scraper._should_scrape_url(url)
            
            # Add to scraped URLs
            scraper.scraped_urls.add(scraper._normalize_url(url))
            
            # Second time should be skipped
            assert not scraper._should_scrape_url(url)
    
    @patch('src.scraper.scraper.BeautifulSoup')
    def test_extract_content(self, mock_soup):
        """Test content extraction from HTML."""
        mock_config = {
            'extraction': {
                'title_selector': 'h1',
                'content_selector': 'main',
                'remove_selectors': ['nav', 'footer'],
                'clean_text': True,
                'min_text_length': 10
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            # Setup mock soup
            mock_soup_instance = Mock()
            mock_soup.return_value = mock_soup_instance
            
            # Mock title element
            mock_title = Mock()
            mock_title.get_text.return_value = "Test Title"
            
            # Mock content element
            mock_content = Mock()
            mock_content.get_text.return_value = "Test content with multiple    spaces\n\n\nand lines"
            
            # Configure select_one to return different elements based on selector
            def mock_select_one(selector):
                if selector == 'h1':
                    return mock_title
                elif selector == 'main':
                    return mock_content
                return None
            
            mock_soup_instance.select_one.side_effect = mock_select_one
            mock_soup_instance.select.return_value = []  # No elements to remove
            
            scraper = WebScraper()
            result = scraper._extract_content("<html>test</html>", "https://example.com")
            
            assert result['title'] == "Test Title"
            assert 'content' in result
            assert len(result['content']) > 0
    
    @patch('src.scraper.scraper.requests.Session')
    @patch('src.scraper.scraper.config')
    def test_scrape_page_success(self, mock_config, mock_session):
        """Test successful page scraping."""
        # Mock the config object properly
        mock_config.scraping.user_agent = "TestBot/1.0"
        mock_config.scraping.timeout_seconds = 30
        mock_config.scraping.max_pages = 100
        mock_config.scraping.delay_seconds = 1
        mock_config.minio = Mock()
        
        mock_site_config = {
            'scraping': {'headers': {'Accept': 'text/html'}},
            'extraction': {
                'title_selector': 'h1',
                'content_selector': 'body',
                'remove_selectors': [],
                'min_text_length': 10,
                'clean_text': True
            },
            'quality': {'skip_if_contains': []}
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_site_config), \
             patch('src.scraper.scraper.MinIOClient'), \
             patch.object(WebScraper, '_extract_content') as mock_extract:
            
            # Setup mocks
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Test content</body></html>"
            mock_response.raise_for_status.return_value = None
            
            mock_session_instance = Mock()
            mock_session_instance.headers = {}  # Make headers assignable
            mock_session_instance.get.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            mock_extract.return_value = {
                'title': 'Test Page',
                'content': 'Test content for the page'
            }
            
            scraper = WebScraper()
            scraper.robot_checker = Mock()
            scraper.robot_checker.can_fetch.return_value = True
            scraper.session = mock_session_instance
            
            result = scraper._scrape_page("https://example.com/test")
            
            assert isinstance(result, ScrapedPage)
            assert result.url == "https://example.com/test"
            assert result.title == "Test Page"
            assert result.status_code == 200
    
    @patch('src.scraper.scraper.requests.Session')
    def test_scrape_page_robots_blocked(self, mock_session):
        """Test page scraping blocked by robots.txt."""
        with patch('src.scraper.scraper.load_site_config'), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            scraper.robot_checker = Mock()
            scraper.robot_checker.can_fetch.return_value = False
            
            result = scraper._scrape_page("https://example.com/blocked")
            
            assert result is None
            assert "https://example.com/blocked" in scraper.failed_urls
    
    @patch('src.scraper.scraper.requests.Session')
    @patch('src.scraper.scraper.config')
    def test_scrape_page_request_error(self, mock_config, mock_session):
        """Test page scraping with request error."""
        # Mock the config object properly
        mock_config.scraping.user_agent = "TestBot/1.0"
        mock_config.scraping.timeout_seconds = 30
        mock_config.minio = Mock()
        
        mock_site_config = {
            'scraping': {'headers': {'Accept': 'text/html'}}
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_site_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            # Setup request error
            mock_session_instance = Mock()
            mock_session_instance.headers = {}  # Make headers assignable
            mock_session_instance.get.side_effect = requests.RequestException("Network error")
            mock_session.return_value = mock_session_instance
            
            scraper = WebScraper()
            scraper.robot_checker = Mock()
            scraper.robot_checker.can_fetch.return_value = True
            scraper.session = mock_session_instance
            
            result = scraper._scrape_page("https://example.com/error")
            
            assert result is None
            assert "https://example.com/error" in scraper.failed_urls
    
    def test_store_page(self):
        """Test storing scraped page to MinIO."""
        with patch('src.scraper.scraper.load_site_config'), \
             patch('src.scraper.scraper.MinIOClient') as mock_minio_client:
            
            # Mock MinIO client
            mock_minio_instance = Mock()
            mock_minio_instance.upload_text.return_value = True
            mock_minio_client.return_value = mock_minio_instance
            
            scraper = WebScraper()
            
            # Create test page
            page = ScrapedPage(
                url="https://example.com/test",
                title="Test Page",
                content="Test content",
                html="<html>Test</html>",
                timestamp=datetime.utcnow(),
                status_code=200,
                response_time=0.5,
                content_length=100
            )
            
            result = scraper._store_page(page)
            
            assert result is True
            # Should call upload_text 3 times (HTML, text, metadata)
            assert mock_minio_instance.upload_text.call_count == 3
    
    def test_scrape_urls_integration(self):
        """Test scraping multiple URLs."""
        mock_config = {
            'urls': {
                'start_urls': ['https://example.com/page1', 'https://example.com/page2'],
                'ignore_patterns': [],
                'follow_patterns': []
            },
            'extraction': {
                'title_selector': 'h1',
                'content_selector': 'body',
                'remove_selectors': [],
                'min_text_length': 10,
                'clean_text': True
            },
            'quality': {'skip_if_contains': []}
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient') as mock_minio, \
             patch('src.scraper.scraper.LineageContext') as mock_lineage, \
             patch('src.scraper.scraper.time.sleep'):  # Speed up test
            
            # Mock MinIO
            mock_minio_instance = Mock()
            mock_minio_instance.ensure_bucket_exists.return_value = True
            mock_minio.return_value = mock_minio_instance
            
            # Mock lineage
            mock_lineage_instance = Mock()
            mock_lineage.return_value.__enter__.return_value = mock_lineage_instance
            
            scraper = WebScraper()
            
            # Mock successful scraping
            with patch.object(scraper, '_scrape_page') as mock_scrape, \
                 patch.object(scraper, '_store_page') as mock_store:
                
                # Mock successful page scraping
                mock_page = Mock(spec=ScrapedPage)
                mock_scrape.return_value = mock_page
                mock_store.return_value = True
                
                urls = ['https://example.com/page1', 'https://example.com/page2']
                stats = scraper.scrape_urls(urls)
                
                assert stats['total_urls'] == 2
                assert 'successful' in stats
                assert 'failed' in stats
                assert 'skipped' in stats
    
    def test_run_method(self):
        """Test the run method."""
        mock_config = {
            'urls': {
                'start_urls': ['https://example.com/page1']
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            
            with patch.object(scraper, 'scrape_urls') as mock_scrape_urls:
                mock_scrape_urls.return_value = {'total_urls': 1, 'successful': 1, 'failed': 0, 'skipped': 0}
                
                result = scraper.run()
                
                assert result['total_urls'] == 1
                mock_scrape_urls.assert_called_once_with(['https://example.com/page1'])
    
    def test_run_no_start_urls(self):
        """Test run method with no start URLs configured."""
        mock_config = {
            'urls': {
                'start_urls': []
            }
        }
        
        with patch('src.scraper.scraper.load_site_config', return_value=mock_config), \
             patch('src.scraper.scraper.MinIOClient'):
            
            scraper = WebScraper()
            
            with pytest.raises(ValueError, match="No start URLs configured"):
                scraper.run()


class TestScraperIntegration:
    """Integration tests for scraper components."""
    
    def test_main_function_structure(self):
        """Test that main function exists and has proper structure."""
        from src.scraper.scraper import main
        
        # Test that main function exists
        assert callable(main)
        
        # Test main with mocked dependencies
        with patch('builtins.print'), \
             patch('src.scraper.scraper.WebScraper') as mock_scraper_class:
            
            mock_scraper = Mock()
            mock_scraper.run.return_value = {'total_urls': 1, 'successful': 1, 'failed': 0, 'skipped': 0}
            mock_scraper_class.return_value = mock_scraper
            
            # Should not raise an exception
            try:
                main()
            except SystemExit:
                pass  # main() might call sys.exit()


if __name__ == "__main__":
    pytest.main([__file__])