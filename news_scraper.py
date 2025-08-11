import argparse
import csv
import json
import logging
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup, Tag

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class ArticleItem:
    def __init__(self) -> None:
        self.url: str = ""
        self.source_name: str = ""
        self.title: str = ""
        self.full_text: str = ""
        self.author: Optional[str] = None
        self.publication_date: Optional[str] = None
        self.scraped_at: str = datetime.now(timezone.utc).isoformat()
        self.spider_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'source_name': self.source_name,
            'title': self.title,
            'full_text': self.full_text,
            'author': self.author,
            'publication_date': self.publication_date,
            'scraped_at': self.scraped_at,
            'spider_name': self.spider_name
        }
    
    def is_valid(self) -> bool:
        essential_fields = [self.url, self.source_name, self.title, self.full_text]
        return all(field and field.strip() for field in essential_fields)


class BaseNewsScraper:
    def __init__(self, name: str, source_name: str, rss_url: str, 
                 allowed_domains: List[str]) -> None:
        self.name = name
        self.source_name = source_name
        self.rss_url = rss_url
        self.allowed_domains = allowed_domains
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Configure requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'zerdisha_scrapers (+https://github.com/awebisam/zerdisha-scrapy)'
        })
    
    def scrape_articles(self, max_articles: int = 50, keywords: Optional[List[str]] = None) -> List[ArticleItem]:
        self.logger.info(f"Starting {self.name} scraper with RSS feed: {self.rss_url}")
        if keywords:
            self.logger.info(f"Filtering articles with keywords: {', '.join(keywords)}")
        
        articles: List[ArticleItem] = []
        
        try:
            feed = feedparser.parse(self.rss_url)
            
            if feed.bozo:
                self.logger.warning(f"RSS feed parsing had issues: {feed.bozo_exception}")
            
            if not hasattr(feed, 'entries') or not feed.entries:
                self.logger.error(f"No entries found in RSS feed: {self.rss_url}")
                return articles
            
            self.logger.info(f"Found {len(feed.entries)} articles in RSS feed")
            
            entries_to_process = feed.entries[:max_articles]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_entry = {
                    executor.submit(self._process_rss_entry, entry): entry 
                    for entry in entries_to_process
                }
                
                for future in as_completed(future_to_entry):
                    entry = future_to_entry[future]
                    try:
                        article = future.result()
                        if article and article.is_valid():
                            if keywords and not self._matches_keywords(article, keywords):
                                self.logger.debug(f"Article filtered out (no keyword match): {article.title[:50]}...")
                                continue
                            
                            articles.append(article)
                            self.logger.info(f"Successfully scraped: {article.title[:50]}...")
                        else:
                            self.logger.warning(f"Failed to extract valid article from {getattr(entry, 'link', 'unknown URL')}")
                    except Exception as e:
                        self.logger.error(f"Error processing entry {getattr(entry, 'link', 'unknown')}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing RSS feed {self.rss_url}: {str(e)}")
        
        self.logger.info(f"Successfully scraped {len(articles)} articles from {self.name}")
        return articles
    
    def _process_rss_entry(self, entry: Any) -> Optional[ArticleItem]:
        if not hasattr(entry, 'link') or not entry.link:
            self.logger.warning("RSS entry missing link, skipping")
            return None
        
        article_url = str(entry.link)
        rss_title = getattr(entry, 'title', '')
        
        parsed_url = urlparse(article_url)
        if not any(domain in parsed_url.netloc for domain in self.allowed_domains):
            self.logger.warning(f"URL not from allowed domains: {article_url}")
            return None
        
        self.logger.debug(f"Processing article: {rss_title[:50]}...")        
        try:
            response = self.session.get(article_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article = self._extract_article_content(soup, article_url, rss_title)
            
            if article:
                self._clean_article(article)
                
            return article
            
        except Exception as e:
            self.logger.error(f"Error fetching/parsing article {article_url}: {str(e)}")
            return None
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str, rss_title: str) -> Optional[ArticleItem]:
        raise NotImplementedError("Subclasses must implement _extract_article_content")
    
    def _clean_article(self, article: ArticleItem) -> None:
        article.title = self._clean_text(article.title)
        article.full_text = self._clean_text(article.full_text)
        article.url = article.url.strip()
        article.source_name = self._clean_text(article.source_name)
        
        if article.author:
            article.author = self._clean_text(article.author)
        
        article.spider_name = self.name
        
        if not article.scraped_at:
            article.scraped_at = datetime.now(timezone.utc).isoformat()
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        cleaned = text.strip()
        cleaned = unicodedata.normalize('NFC', cleaned)
        
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _extract_publication_date(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        try:
            meta_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="publish-date"]',
                'meta[name="date"]'
            ]
            
            for selector in meta_selectors:
                meta_tag = soup.select_one(selector)
                if meta_tag and meta_tag.get('content'):
                    try:
                        date_str = meta_tag['content']
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            time_selectors = [
                'time[datetime]',
                '.published-date time',
                '.post-date time',
                '.entry-date time'
            ]
            
            for selector in time_selectors:
                time_element = soup.select_one(selector)
                if time_element and time_element.get('datetime'):
                    try:
                        date_str = time_element['datetime']
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            date_selectors = [
                '.published-date', '.post-date', '.entry-date',
                '.article-date', '.date', '.timestamp'
            ]
            
            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    date_text = date_element.get_text().strip()
                    parsed_date = self._parse_date_text(date_text)
                    if parsed_date:
                        return parsed_date
            
            return self._extract_date_from_url(url)
            
        except Exception as e:
            self.logger.error(f"Error extracting publication date from {url}: {str(e)}")
            return None
    
    def _parse_date_text(self, date_text: str) -> Optional[str]:
        if not date_text:
            return None
        
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})', 
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                date_str = match.group(1)
                formats = ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]
                
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
        
        return None
    
    def _extract_date_from_url(self, url: str) -> Optional[str]:
        url_patterns = [
            r'/(\d{4})/(\d{2})/(\d{2})/',
            r'/(\d{4}-\d{2}-\d{2})/',
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    if len(match.groups()) == 3:
                        year, month, day = match.groups()
                        date_str = f"{year}-{month}-{day}"
                    else:
                        date_str = match.group(1)
                    
                    # Validate the date
                    datetime.strptime(date_str, "%Y-%m-%d")
                    return date_str
                except ValueError:
                    continue
        
        return None
    
    def _matches_keywords(self, article: ArticleItem, keywords: List[str]) -> bool:
        if not keywords:
            return True
        
        searchable_text = f"{article.title} {article.full_text}".lower()
        
        for keyword in keywords:
            if keyword.lower() in searchable_text:
                self.logger.debug(f"Keyword '{keyword}' found in article: {article.title[:50]}...")
                return True
        
        return False


class KathmanduPostScraper(BaseNewsScraper):    
    def __init__(self) -> None:
        super().__init__(
            name="kathmandupost",
            source_name="The Kathmandu Post",
            rss_url="https://kathmandupost.com/rss",
            allowed_domains=["kathmandupost.com"]
        )
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str, rss_title: str) -> Optional[ArticleItem]:
        try:
            paragraphs = soup.select('main p')
            
            if not paragraphs:
                self.logger.warning(f"No content found using CSS selector 'main p' for {url}")
                return None            
            paragraph_texts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    paragraph_texts.append(text)
            
            if not paragraph_texts:
                self.logger.warning(f"No meaningful content extracted from {url}")
                return None
            
            full_text = '\n\n'.join(paragraph_texts)
            
            title = rss_title
            if not title:
                title_element = soup.select_one('h1')
                title = title_element.get_text().strip() if title_element else ''
            
            if not title:
                self.logger.warning(f"No title found for {url}")
                return None
            
            author = None
            author_element = soup.select_one('.article-author')
            if author_element:
                author = author_element.get_text().strip()
            
            publication_date = self._extract_publication_date_kathmandupost(soup, url)
            
            article = ArticleItem()
            article.url = url
            article.source_name = self.source_name
            article.title = title
            article.full_text = full_text
            article.author = author
            article.publication_date = publication_date
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def _extract_publication_date_kathmandupost(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        try:
            published_elements = soup.select('.updated-time')
            for element in published_elements:
                text = element.get_text()
                if "Published at" in text:
                    date_part = text.split("Published at")[1].strip()
                    if date_part.startswith(":"):
                        date_part = date_part[1:].strip()
                    
                    try:
                        parsed_date = datetime.strptime(date_part, "%B %d, %Y")
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            return self._extract_publication_date(soup, url)
            
        except Exception as e:
            self.logger.debug(f"Error in Kathmandu Post specific date extraction: {str(e)}")
            return self._extract_publication_date(soup, url)


class AnnapurnaScraper(BaseNewsScraper):
    
    def __init__(self) -> None:
        super().__init__(
            name="annapurna",
            source_name="The Annapurna Express",
            rss_url="https://theannapurnaexpress.com/rss/",
            allowed_domains=["theannapurnaexpress.com"]
        )
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str, rss_title: str) -> Optional[ArticleItem]:
        try:
            paragraphs = soup.select('.detail__page-content p')
            
            if not paragraphs:
                self.logger.warning(f"No content found using CSS selector '.detail__page-content p' for {url}")
                return None
            
            paragraph_texts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    paragraph_texts.append(text)
            
            if not paragraph_texts:
                self.logger.warning(f"No meaningful content extracted from {url}")
                return None
            
            full_text = '\n\n'.join(paragraph_texts)
            
            title = rss_title
            if not title:
                title_element = soup.select_one('h1.single-title')
                title = title_element.get_text().strip() if title_element else ''
            
            if not title:
                self.logger.warning(f"No title found for {url}")
                return None
            
            author = None
            author_selectors = ['.author-name', 'span.byline']
            for selector in author_selectors:
                author_element = soup.select_one(selector)
                if author_element:
                    author = author_element.get_text().strip()
                    break
            
            publication_date = self._extract_publication_date(soup, url)
            
            article = ArticleItem()
            article.url = url
            article.source_name = self.source_name
            article.title = title
            article.full_text = full_text
            article.author = author
            article.publication_date = publication_date
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None


class NagarikScraper(BaseNewsScraper):
    
    def __init__(self) -> None:
        super().__init__(
            name="nagarik",
            source_name="Nagarik News",
            rss_url="https://nagariknews.nagariknetwork.com/feed",
            allowed_domains=["nagariknetwork.com"]
        )
    
    def _extract_article_content(self, soup: BeautifulSoup, url: str, rss_title: str) -> Optional[ArticleItem]:
        try:
            paragraphs = soup.select('.content-wrapper p')
            
            if not paragraphs:
                all_paragraphs = soup.select('p')
                paragraphs = []
                
                for p in all_paragraphs:
                    text = p.get_text().strip()
                    if (text and len(text) > 50 and 
                        'नागरिक अभिलेखालय' not in text and 
                        'Please Enable javascript' not in text and
                        'Subscribe' not in text):
                        paragraphs.append(p)
            
            if not paragraphs:
                self.logger.warning(f"No content found for {url}")
                return None
            paragraph_texts = []
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    paragraph_texts.append(text)
            
            if not paragraph_texts:
                self.logger.warning(f"No meaningful content extracted from {url}")
                return None
            
            full_text = '\n\n'.join(paragraph_texts)
            
            title = rss_title
            if not title:
                title_element = soup.select_one('h1')
                title = title_element.get_text().strip() if title_element else ''
            
            if not title:
                self.logger.warning(f"No title found for {url}")
                return None
            
            author = None
            author_element = soup.select_one('.article-author')
            if author_element:
                author = author_element.get_text().strip()
            
            publication_date = self._extract_publication_date(soup, url)
            
            article = ArticleItem()
            article.url = url
            article.source_name = self.source_name
            article.title = title
            article.full_text = full_text
            article.author = author
            article.publication_date = publication_date
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None


class UnifiedNewsScraper:
    
    def __init__(self, output_dir: str = "data", output_format: str = "both", 
                 keywords: Optional[List[str]] = None) -> None:
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.keywords = keywords
        self.scrapers = {
            'kathmandupost': KathmanduPostScraper(),
            'annapurna': AnnapurnaScraper(),
            'nagarik': NagarikScraper()
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def scrape_all_sources(self, max_articles_per_source: int = 50) -> Dict[str, List[ArticleItem]]:
        
        all_articles = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(scraper.scrape_articles, max_articles_per_source, self.keywords): source
                for source, scraper in self.scrapers.items()
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    articles = future.result()
                    all_articles[source] = articles
                    logger.info(f"Completed scraping {len(articles)} articles from {source}")
                except Exception as e:
                    logger.error(f"Error scraping {source}: {str(e)}")
                    all_articles[source] = []
        
        return all_articles
    
    def scrape_sources(self, sources: List[str], max_articles_per_source: int = 50) -> Dict[str, List[ArticleItem]]:
        logger.info(f"Starting scraping for sources: {', '.join(sources)}")
        
        all_articles = {}
        
        selected_scrapers = {
            source: scraper for source, scraper in self.scrapers.items() 
            if source in sources
        }
        
        if not selected_scrapers:
            logger.error(f"No valid sources found. Available sources: {list(self.scrapers.keys())}")
            return all_articles
        
        with ThreadPoolExecutor(max_workers=len(selected_scrapers)) as executor:
            future_to_source = {
                executor.submit(scraper.scrape_articles, max_articles_per_source, self.keywords): source
                for source, scraper in selected_scrapers.items()
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    articles = future.result()
                    all_articles[source] = articles
                    logger.info(f"Completed scraping {len(articles)} articles from {source}")
                except Exception as e:
                    logger.error(f"Error scraping {source}: {str(e)}")
                    all_articles[source] = []
        
        return all_articles
    
    def save_articles(self, all_articles: Dict[str, List[ArticleItem]]) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for source, articles in all_articles.items():
            if not articles:
                logger.warning(f"No articles to save for {source}")
                continue
            
            articles_data = [article.to_dict() for article in articles]
            
            if self.output_format in ('json', 'both'):
                json_filename = f"articles_{source}_{timestamp}.json"
                json_path = os.path.join(self.output_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(articles_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(articles)} articles from {source} to {json_path}")
            
            if self.output_format in ('csv', 'both'):
                csv_filename = f"articles_{source}_{timestamp}.csv"
                csv_path = os.path.join(self.output_dir, csv_filename)
                
                if articles_data:
                    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=articles_data[0].keys())
                        writer.writeheader()
                        writer.writerows(articles_data)
                    
                    logger.info(f"Saved {len(articles)} articles from {source} to {csv_path}")
        
        all_articles_combined = []
        for articles in all_articles.values():
            all_articles_combined.extend(article.to_dict() for article in articles)
        
        if all_articles_combined:
            if self.output_format in ('json', 'both'):
                unified_json_path = os.path.join(self.output_dir, f"all_articles_unified_{timestamp}.json")
                with open(unified_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_articles_combined, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(all_articles_combined)} unified articles to {unified_json_path}")
            
            if self.output_format in ('csv', 'both'):
                unified_csv_path = os.path.join(self.output_dir, f"all_articles_unified_{timestamp}.csv")
                with open(unified_csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=all_articles_combined[0].keys())
                    writer.writeheader()
                    writer.writerows(all_articles_combined)
                logger.info(f"Saved {len(all_articles_combined)} unified articles to {unified_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified News Scraper for Nepali News Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_news_scraper.py
  python unified_news_scraper.py --sources kathmandupost,nagarik
  python unified_news_scraper.py --output-dir data --format json --max-articles 30
  python unified_news_scraper.py --keywords "KP Oli,government,politics"
  python unified_news_scraper.py --sources kathmandupost --keywords "Nepal,economy" --max-articles 20
        """
    )
    
    parser.add_argument(
        '--sources',
        type=str,
        default='kathmandupost,annapurna,nagarik',
        help='Comma-separated list of sources to scrape (default: all sources)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for scraped articles (default: data)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    
    parser.add_argument(
        '--max-articles',
        type=int,
        default=50,
        help='Maximum articles to scrape per source (default: 50)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--keywords',
        type=str,
        help='Comma-separated list of keywords to filter articles (e.g., "KP Oli,government,politics")'
    )
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    
    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',') if k.strip()]
        logger.info(f"Filtering articles with keywords: {', '.join(keywords)}")
    
    scraper = UnifiedNewsScraper(
        output_dir=args.output_dir,
        output_format=args.format,
        keywords=keywords
    )
    
    available_sources = list(scraper.scrapers.keys())
    invalid_sources = [s for s in sources if s not in available_sources]
    if invalid_sources:
        logger.error(f"Invalid sources: {invalid_sources}. Available sources: {available_sources}")
        sys.exit(1)
    
    try:
        if set(sources) == set(available_sources):
            all_articles = scraper.scrape_all_sources(args.max_articles)
        else:
            all_articles = scraper.scrape_sources(sources, args.max_articles)
        
        scraper.save_articles(all_articles)
        
        total_articles = sum(len(articles) for articles in all_articles.values())
        if keywords:
            logger.info(f"Successfully scraped {total_articles} articles total (filtered by keywords: {', '.join(keywords)})")
        else:
            logger.info(f"Successfully scraped {total_articles} articles total")
        
        for source, articles in all_articles.items():
            logger.info(f"  {source}: {len(articles)} articles")
    
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
