#!/usr/bin/env python3
"""
Enhanced Database Manager Module - FINAL VERSION
Secure PostgreSQL connection management with a high-performance, two-step query strategy.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date # ANNOTATION: 'date' is now imported
from contextlib import asynccontextmanager
from functools import lru_cache

import asyncpg
from config_app_config import get_config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom database error"""
    pass

class QueryTimeoutError(DatabaseError):
    """Query timeout error"""
    pass

class ConnectionPoolError(DatabaseError):
    """Connection pool error"""
    pass

class DatabaseManager:
    """Enhanced PostgreSQL database manager with a high-performance, two-step search strategy."""
    
    def __init__(self, config=None):
        self.config = config or get_config().database
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._stats = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'connection_errors': 0,
            'last_health_check': None,
        }
    
    async def initialize(self) -> bool:
        """Initialize database connection pool."""
        if self._initialized: return True
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host, port=self.config.port, database=self.config.database,
                user=self.config.username, password=self.config.password,
                min_size=self.config.min_connections, max_size=self.config.max_connections,
                command_timeout=self.config.query_timeout,
                server_settings={'default_transaction_read_only': 'on' if self.config.read_only_mode else 'off'}
            )
            self._initialized = True      
            logger.info(f"✓ Database manager initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise DatabaseError(f"Database connection failed: {str(e)}")
    
    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("✓ Database pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self._initialized or not self._pool:
            raise ConnectionPoolError("Database not initialized")
        conn = None
        try:
            conn = await self._pool.acquire()
            yield conn
        finally:
            if conn:
                await self._pool.release(conn)
    
    async def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute SQL query."""
        query_start = time.time()
        try:
            async with self.get_connection() as conn:
                result = await conn.fetch(query, *params) if params else await conn.fetch(query)
                duration = time.time() - query_start
                if duration > 10.0:
                    logger.warning(f"SLOW QUERY DETECTED: {duration:.2f}s")
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

    async def _find_matching_serials(self, variations: List[str], limit: int, stage: str) -> List[str]:
        """STEP 1: Quickly find only the serial numbers of matching trademarks."""
        try:
            if stage == 'exact':
                condition = "UPPER(mark_identification) = ANY($1)"
                patterns = variations
            elif stage == 'starts':
                condition = "UPPER(mark_identification) LIKE ANY($1)"
                patterns = [f"{var}%" for var in variations if len(var) >= 3]
            elif stage == 'contains':
                condition = "UPPER(mark_identification) LIKE ANY($1)"
                patterns = [f"%{var}%" for var in variations if len(var) >= 4]
            else:
                return []
            
            if not patterns:
                return []
            
            query = f"""
                SELECT serial_number
                FROM case_files
                WHERE {condition}
                LIMIT $2
            """
            params = (patterns, limit)
            results = await self.execute_query(query, params)
            return [row['serial_number'] for row in results]
        except Exception as e:
            logger.warning(f"Serial number search failed for stage '{stage}': {str(e)}")
            return []

    async def _enrich_serials(self, serials: List[str]) -> List[Dict[str, Any]]:
        """STEP 2: Take a small list of serial numbers and gather all their details."""
        if not serials:
            return []
        
        # ANNOTATION: The join condition for tm_status_codes is changed from cf.status_code::text 
        # to cf.status_code::integer to resolve the data type mismatch error.
        query = """
            SELECT
                cf.serial_number,
                cf.mark_identification,
                cf.status_code,
                COALESCE(cf.filing_date, '0001-01-01'::date) as filing_date,
                COALESCE(cf.registration_date, '0001-01-01'::date) as registration_date,
                cfo.owner,
                cls.nice_classes,
                CONCAT(tsc.status_description, ' (', COALESCE(tsc.status_detail, 'General'), ')') AS full_status_description
            FROM case_files cf
            LEFT JOIN (
                SELECT DISTINCT ON (serial_number) serial_number, party_name as owner
                FROM case_file_owners
                WHERE serial_number = ANY($1)
            ) AS cfo ON cf.serial_number = cfo.serial_number
            LEFT JOIN (
                SELECT serial_number, STRING_AGG(DISTINCT international_code, ',' ORDER BY international_code) as nice_classes
                FROM classifications
                WHERE serial_number = ANY($1)
                GROUP BY serial_number
            ) AS cls ON cf.serial_number = cls.serial_number
            LEFT JOIN tm_status_codes tsc ON cf.status_code::integer = tsc.status_code
            WHERE cf.serial_number = ANY($1)
        """
        params = (serials,)
        return await self.execute_query(query, params)

    async def _run_search_stage(self, variations: List[str], limit: int, stage: str) -> List[Dict[str, Any]]:
        """Orchestrates the new two-step query process for a single search stage."""
        matching_serials = await self._find_matching_serials(variations, limit, stage)
        enriched_results = await self._enrich_serials(matching_serials)
        return self._process_search_results(enriched_results)

    async def search_trademarks_basic(self, variations: List[str], limit: int) -> List[Dict[str, Any]]:
        """Basic search performs an exact match search for all name variations."""
        return await self._run_search_stage(variations, limit, 'exact')

    async def search_trademarks_advanced(self, variations: List[str], limit: int) -> List[Dict[str, Any]]:
        """
        Performs a multi-stage advanced search.
        """
        # ANNOTATION: This function is completely reworked to prioritize results by relevance, not date.
        all_serials = set()
        combined_results = []

        # Stage 1: Get the most relevant exact matches first.
        exact_results = await self._run_search_stage(variations, limit, 'exact')
        for result in exact_results:
            serial = result.get('serial_number')
            if serial and serial not in all_serials:
                all_serials.add(serial)
                combined_results.append(result)

        # Stage 2: Fill with "starts with" matches if there's still room.
        if len(combined_results) < limit:
            starts_limit = limit - len(combined_results)
            starts_results = await self._run_search_stage(variations, starts_limit, 'starts')
            for result in starts_results:
                serial = result.get('serial_number')
                if serial and serial not in all_serials:
                    all_serials.add(serial)
                    combined_results.append(result)
        
        # Stage 3: Fill any remaining space with "contains" matches.
        if len(combined_results) < limit:
            contains_limit = limit - len(combined_results)
            contains_results = await self._run_search_stage(variations, contains_limit, 'contains')
            for result in contains_results:
                serial = result.get('serial_number')
                if serial and serial not in all_serials:
                    all_serials.add(serial)
                    combined_results.append(result)

        return combined_results[:limit]
            
    def _process_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper to process results, e.g., converting class strings to lists."""
        for result in results:
            classes_str = result.get('nice_classes')
            if isinstance(classes_str, str):
                result['nice_classes'] = classes_str.split(',') if classes_str else []
            elif classes_str is None:
                result['nice_classes'] = []
        return results

    async def search_trademarks_fuzzy(self, patterns: List[str], limit: int) -> List[Dict[str, Any]]:
        """Legacy fuzzy search, now maps to the 'contains' search stage."""
        logger.info("Using legacy fuzzy search method (now 'contains' stage).")
        return await self._run_search_stage(patterns, limit, 'contains')
