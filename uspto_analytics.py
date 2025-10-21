# uspto_analytics.py
import logging
import pandas as pd
from database_manager import DatabaseManager
from typing import Dict, Any

import asyncio

logger = logging.getLogger(__name__)

async def get_global_analytics(db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Fetches and calculates global USPTO analytics data for the main dashboard.
    """
    if not db_manager:
        logger.error("Database manager is not initialized for analytics.")
        return {"error": "Database manager not available."}

    try:
        # Query 1: Historical filings by year (since 2000)
        filings_by_year_query = """
            SELECT EXTRACT(YEAR FROM filing_date)::int AS year, COUNT(serial_number) AS count
            FROM case_files WHERE filing_date IS NOT NULL AND filing_date >= '2000-01-01'
            GROUP BY year ORDER BY year DESC;
        """
        
        # Query 2: Top 3 Filing Organizations (Owners)
        top_owners_query = """
            SELECT owner, COUNT(serial_number) AS count
            FROM case_file_owners
            GROUP BY owner
            ORDER BY count DESC
            LIMIT 3;
        """

        # Query 3: Top 3 Filing Firms (Correspondents)
        top_firms_query = """
            SELECT correspondent_name AS firm, COUNT(serial_number) AS count
            FROM case_file_correspondents
            GROUP BY firm
            ORDER BY count DESC
            LIMIT 3;
        """

        # Query 4: Status distribution
        status_dist_query = """
            SELECT live_dead_indicator as status, COUNT(serial_number) as count
            FROM case_files GROUP BY status;
        """

        # Query 5: Processing time (since 2000)
        processing_time_query = """
            SELECT
                EXTRACT(YEAR FROM filing_date)::int as year,
                AVG(registration_date - filing_date) as avg_days
            FROM case_files
            WHERE registration_date IS NOT NULL AND filing_date IS NOT NULL
              AND registration_date > filing_date AND filing_date >= '2000-01-01'
            GROUP BY year ORDER BY year DESC;
        """

        # Execute all queries in parallel for efficiency
        queries = {
            "filings": db_manager.execute_query(filings_by_year_query),
            "top_owners": db_manager.execute_query(top_owners_query),
            "top_firms": db_manager.execute_query(top_firms_query),
            "status_dist": db_manager.execute_query(status_dist_query),
            "processing_time": db_manager.execute_query(processing_time_query)
        }
        results = await asyncio.gather(*queries.values())
        results_dict = dict(zip(queries.keys(), results))

        # Format data for JSON response
        analytics_data = {
            "historical_overview": {
                "applications_by_year": {row['year']: row['count'] for row in results_dict["filings"]}
            },
            "top_filers": {
                "top_organizations": {row['owner']: row['count'] for row in results_dict["top_owners"]},
                "top_firms": {row['firm']: row['count'] for row in results_dict["top_firms"]},
            },
            "status_distribution": {
                "current_distribution": {row['status']: row['count'] for row in results_dict["status_dist"]}
            },
            "timeline_analysis": {
                "avg_days_to_registration": {row['year']: int(row['avg_days']) for row in results_dict["processing_time"] if row['avg_days'] is not None}
            }
        }
        return analytics_data

    except Exception as e:
        logger.error(f"Error generating global analytics: {e}")
        return {"error": str(e)}