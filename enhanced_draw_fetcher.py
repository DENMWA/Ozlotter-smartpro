"""
Enhanced draw fetcher with robust error handling and multiple data sources
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import random
from typing import Optional, List, Dict
from config import *
from utils import setup_logger

logger = setup_logger(__name__)

class EnhancedDrawFetcher:
    def __init__(self):
        self.save_path = DRAWS_FILE
        self.backup_sources = [
            "https://www.lotto.net/oz-lotto/numbers/",
            "https://www.thelott.com/oz-lotto/results",
            "https://www.lottoland.com.au/oz-lotto/results"
        ]
        
    def fetch_draws_with_retry(self, limit: int = 300, max_retries: int = 3) -> pd.DataFrame:
        """Fetch draws with retry logic and multiple sources"""
        logger.info(f"Fetching up to {limit} draws with {max_retries} retries...")
        
        for attempt in range(max_retries):
            try:
                draws_df = self._fetch_from_primary_source(limit)
                
                if not draws_df.empty:
                    logger.info(f"Successfully fetched {len(draws_df)} draws on attempt {attempt + 1}")
                    self._save_draws(draws_df)
                    return draws_df
                
                logger.warning(f"Primary source failed on attempt {attempt + 1}")
                
                for backup_url in self.backup_sources:
                    try:
                        draws_df = self._fetch_from_backup_source(backup_url, limit)
                        if not draws_df.empty:
                            logger.info(f"Successfully fetched from backup source: {backup_url}")
                            self._save_draws(draws_df)
                            return draws_df
                    except Exception as e:
                        logger.warning(f"Backup source {backup_url} failed: {e}")
                        continue
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        
        logger.warning("All fetch attempts failed, loading local data...")
        return self.load_local_draws()
    
    def _fetch_from_primary_source(self, limit: int) -> pd.DataFrame:
        """Fetch from primary lotto.net source"""
        draws = []
        page = 1
        
        while len(draws) < limit:
            url = f"https://www.lotto.net/oz-lotto/numbers/{page}"
            
            response = self._make_request(url)
            if not response:
                break
            
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select(".resultsTable .resultsRow")
            
            if not rows:
                rows = soup.select(".results-table tr")
                if not rows:
                    rows = soup.select("table tr")
            
            page_draws = 0
            for row in rows:
                balls = row.select(".balls .ball")
                if not balls:
                    balls = row.select(".ball")
                    if not balls:
                        balls = row.select("td")
                
                if len(balls) >= 7:
                    try:
                        nums = []
                        for ball in balls[:7]:
                            text = ball.text.strip()
                            if text.isdigit():
                                num = int(text)
                                if MIN_NUMBER <= num <= MAX_NUMBER:
                                    nums.append(num)
                        
                        if len(nums) == 7:
                            draws.append(nums)
                            page_draws += 1
                            
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Error parsing row: {e}")
                        continue
                
                if len(draws) >= limit:
                    break
            
            if page_draws == 0:
                logger.warning(f"No draws found on page {page}")
                break
                
            page += 1
            
            time.sleep(random.uniform(0.5, 1.5))
        
        if draws:
            df = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(7)])
            return df
        
        return pd.DataFrame()
    
    def _fetch_from_backup_source(self, url: str, limit: int) -> pd.DataFrame:
        """Fetch from backup source with generic parsing"""
        response = self._make_request(url)
        if not response:
            return pd.DataFrame()
        
        soup = BeautifulSoup(response.text, "html.parser")
        draws = []
        
        number_patterns = soup.find_all(string=lambda text: text and text.strip().isdigit())
        
        current_draw = []
        for text in number_patterns:
            try:
                num = int(text.strip())
                if MIN_NUMBER <= num <= MAX_NUMBER:
                    current_draw.append(num)
                    
                    if len(current_draw) == 7:
                        draws.append(current_draw[:])
                        current_draw = []
                        
                        if len(draws) >= limit:
                            break
            except ValueError:
                continue
        
        if draws:
            df = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(7)])
            return df
        
        return pd.DataFrame()
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with proper headers and error handling"""
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def _save_draws(self, draws_df: pd.DataFrame):
        """Save draws to file with backup"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            
            if os.path.exists(self.save_path):
                backup_path = f"{self.save_path}.backup"
                os.rename(self.save_path, backup_path)
            
            draws_df.to_csv(self.save_path, index=False)
            logger.info(f"Saved {len(draws_df)} draws to {self.save_path}")
            
        except Exception as e:
            logger.error(f"Error saving draws: {e}")
    
    def load_local_draws(self) -> pd.DataFrame:
        """Load locally cached historical draws with validation"""
        try:
            if os.path.exists(self.save_path):
                df = pd.read_csv(self.save_path)
                
                if self._validate_draws_data(df):
                    logger.info(f"Loaded {len(df)} draws from local cache")
                    return df
                else:
                    logger.warning("Local draws data failed validation")
            
            backup_path = f"{self.save_path}.backup"
            if os.path.exists(backup_path):
                df = pd.read_csv(backup_path)
                if self._validate_draws_data(df):
                    logger.info(f"Loaded {len(df)} draws from backup")
                    return df
            
        except Exception as e:
            logger.error(f"Error loading local draws: {e}")
        
        return pd.DataFrame()
    
    def _validate_draws_data(self, df: pd.DataFrame) -> bool:
        """Validate draws data format and content"""
        if df.empty:
            return False
        
        if len(df.columns) != 7:
            logger.warning(f"Expected 7 columns, got {len(df.columns)}")
            return False
        
        for _, row in df.head(10).iterrows():  # Check first 10 rows
            for value in row.values:
                try:
                    num = int(value)
                    if not (MIN_NUMBER <= num <= MAX_NUMBER):
                        logger.warning(f"Number {num} out of range [{MIN_NUMBER}, {MAX_NUMBER}]")
                        return False
                except (ValueError, TypeError):
                    logger.warning(f"Invalid number format: {value}")
                    return False
        
        for _, row in df.head(10).iterrows():
            numbers = list(row.values)
            if len(set(numbers)) != len(numbers):
                logger.warning("Found duplicate numbers within a draw")
                return False
        
        return True
    
    def get_data_freshness(self) -> Dict[str, str]:
        """Get information about data freshness"""
        info = {
            'local_file_exists': os.path.exists(self.save_path),
            'backup_file_exists': os.path.exists(f"{self.save_path}.backup"),
            'last_modified': 'Unknown',
            'record_count': 0
        }
        
        try:
            if info['local_file_exists']:
                stat = os.stat(self.save_path)
                info['last_modified'] = time.ctime(stat.st_mtime)
                
                df = pd.read_csv(self.save_path)
                info['record_count'] = len(df)
        except Exception as e:
            logger.warning(f"Error getting data freshness: {e}")
        
        return info
