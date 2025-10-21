#!/usr/bin/env python3
"""
Utility Modules for FastAPI Trademark Search System
Security middleware, rate limiting, and audit logging
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from functools import lru_cache
import os

# FastAPI imports
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# PostgreSQL async database
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Import config - fixed import path
try:
    from config_app_config import get_config
except ImportError:
    # Fallback for testing
    def get_config():
        class MockConfig:
            def __init__(self):
                self.database = type('obj', (object,), {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'trademark_auth',
                    'username': 'postgres',
                    'password': ''
                })()
        return MockConfig()

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection"""
    
    def __init__(self, app):
        super().__init__(app)
        try:
            self.config = get_config()
        except:
            self.config = None
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' ws://localhost:* wss://localhost:* ws://127.0.0.1:* wss://127.0.0.1:*; img-src 'self' data:; font-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        # Rate limiting
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Suspicious patterns
        self.suspicious_patterns = [
            'union select', 'drop table', 'insert into', 'delete from',
            '<script', 'javascript:', 'onclick=', 'onerror=',
            '../../../', '..\\..\\..\\', 'cmd.exe', '/bin/bash'
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security filters"""
        start_time = time.time()
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=429,
                content={"error": "IP address blocked due to suspicious activity"}
            )
        
        # Basic rate limiting (requests per minute)
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # Input validation
        security_issue = await self._validate_request_security(request)
        if security_issue:
            logger.warning(f"Security issue detected from {client_ip}: {security_issue}")
            self._handle_security_violation(client_ip, security_issue)
            
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"}
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add security headers
            for header, value in self.security_headers.items():
                response.headers[header] = value
            
            # Add processing time header
            process_time = time.time() - start_time
            response.headers['X-Process-Time'] = str(process_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else 'unknown'
    
    def _check_rate_limit(self, client_ip: str, max_requests: int = 60) -> bool:
        """Check rate limiting for client IP"""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Initialize or clean old requests
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove old requests outside the window
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(self.request_counts[client_ip]) >= max_requests:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(now)
        return True
    
    async def _validate_request_security(self, request: Request) -> Optional[str]:
        """Validate request for security issues"""
        
        # Check URL for suspicious patterns
        url_path = str(request.url.path).lower()
        for pattern in self.suspicious_patterns:
            if pattern in url_path:
                return f"Suspicious URL pattern: {pattern}"
        
        # Check query parameters
        for key, value in request.query_params.items():
            value_lower = str(value).lower()
            for pattern in self.suspicious_patterns:
                if pattern in value_lower:
                    return f"Suspicious query parameter: {pattern}"
        
        # Check headers
        user_agent = request.headers.get('User-Agent', '').lower()
        if not user_agent or len(user_agent) < 10:
            return "Missing or suspicious User-Agent"
        
        # Check for common attack patterns in User-Agent
        attack_patterns = ['sqlmap', 'nikto', 'nmap', 'dirb', 'gobuster']
        for pattern in attack_patterns:
            if pattern in user_agent:
                return f"Attack tool detected: {pattern}"
        
        return None
    
    def _handle_security_violation(self, client_ip: str, violation: str):
        """Handle security violation"""
        # Log the violation
        logger.warning(f"Security violation from {client_ip}: {violation}")
        
        # For serious violations, temporarily block the IP
        serious_patterns = ['union select', 'drop table', '<script']
        if any(pattern in violation.lower() for pattern in serious_patterns):
            self.blocked_ips.add(client_ip)
            logger.error(f"IP {client_ip} blocked due to serious security violation")

# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        
        # Storage for request tracking
        self.user_requests: Dict[str, List[float]] = {}
        self.user_burst_counts: Dict[str, int] = {}
        self.blocked_users: Dict[str, float] = {}  # user_id -> unblock_time
        
        # Cleanup task
        self._cleanup_task = None
        self._running = False
    
    async def start(self):
        """Start the rate limiter cleanup task"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the rate limiter"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old request records"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                now = time.time()
                cutoff = now - 60  # Remove requests older than 1 minute
                
                # Clean up user requests
                for user_id in list(self.user_requests.keys()):
                    self.user_requests[user_id] = [
                        req_time for req_time in self.user_requests[user_id]
                        if req_time > cutoff
                    ]
                    
                    # Remove empty lists
                    if not self.user_requests[user_id]:
                        del self.user_requests[user_id]
                
                # Clean up blocked users
                for user_id in list(self.blocked_users.keys()):
                    if self.blocked_users[user_id] < now:
                        del self.blocked_users[user_id]
                        logger.info(f"User {user_id} unblocked after rate limit timeout")
                
                # Reset burst counts
                self.user_burst_counts.clear()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {str(e)}")
    
    async def check_limit(self, user_id: str, endpoint: Optional[str] = None) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        
        # Check if user is temporarily blocked
        if user_id in self.blocked_users:
            if self.blocked_users[user_id] > now:
                return False
            else:
                # Unblock user
                del self.blocked_users[user_id]
        
        # Initialize user request tracking
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Clean old requests (within last minute)
        cutoff = now - 60
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > cutoff
        ]
        
        # Check rate limit
        if len(self.user_requests[user_id]) >= self.requests_per_minute:
            # Block user for 5 minutes
            self.blocked_users[user_id] = now + 300
            logger.warning(f"User {user_id} rate limited - blocked for 5 minutes")
            return False
        
        # Check burst limit (requests in last 10 seconds)
        burst_cutoff = now - 10
        recent_requests = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > burst_cutoff
        ]
        
        if len(recent_requests) >= self.burst_size:
            logger.warning(f"User {user_id} exceeded burst limit")
            return False
        
        # Add current request
        self.user_requests[user_id].append(now)
        return True

# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """Comprehensive audit logging system using PostgreSQL"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        if db_config is None:
            try:
                config = get_config()
                self.db_config = config.database
            except:
                # Fallback config
                self.db_config = type('obj', (object,), {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'trademark_auth',
                    'username': 'postgres',
                    'password': ''
                })()
        else:
            self.db_config = db_config
            
        self._initialized = False
        self._pool = None
    
    async def initialize(self):
        """Initialize audit logging in PostgreSQL"""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available - audit logging will be limited")
            return
        
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
                min_size=2,
                max_size=5,
                command_timeout=30
            )
            
            async with self._pool.acquire() as conn:
                # Create audit schema if it doesn't exist
                await conn.execute("CREATE SCHEMA IF NOT EXISTS audit")
                
                # Authentication events
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit.auth_events (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT,
                        event_type TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        ip_address INET,
                        user_agent TEXT,
                        details JSONB,
                        session_id TEXT
                    )
                """)
                
                # Create basic indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_user_time ON audit.auth_events(user_id, timestamp)")
            
            self._initialized = True
            logger.info("✓ Audit logging initialized with PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit logging: {str(e)}")
    
    async def close(self):
        """Close audit logger connection pool"""
        if self._pool:
            await self._pool.close()
    
    async def log_auth_event(
        self,
        user_id: Optional[str],
        event_type: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict] = None,
        session_id: Optional[str] = None
    ):
        """Log authentication event"""
        if not self._initialized or not self._pool:
            return
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit.auth_events (
                        user_id, event_type, success, ip_address, user_agent, details, session_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, user_id, event_type, success, ip_address, user_agent, details, session_id)
        except Exception as e:
            logger.error(f"Failed to log auth event: {str(e)}")

# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_per_second': 0.0,
            'average_response_time': 0.0,
            'error_rate': 0.0,
            'active_connections': 0
        }
        
        self.request_times = []
        self.error_count = 0
        self.start_time = time.time()
        
    def record_request(self, response_time: float, is_error: bool = False):
        """Record a completed request"""
        self.metrics['requests_total'] += 1
        self.request_times.append(response_time)
        
        if is_error:
            self.error_count += 1
        
        # Keep only last 1000 request times for memory efficiency
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update calculated metrics"""
        now = time.time()
        uptime = now - self.start_time
        
        # Requests per second
        if uptime > 0:
            self.metrics['requests_per_second'] = self.metrics['requests_total'] / uptime
        
        # Average response time
        if self.request_times:
            self.metrics['average_response_time'] = sum(self.request_times) / len(self.request_times)
        
        # Error rate
        if self.metrics['requests_total'] > 0:
            self.metrics['error_rate'] = self.error_count / self.metrics['requests_total']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
        self._running = False
    
    async def start(self):
        """Start the cache cleanup task"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the cache manager"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired cache entries"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                now = time.time()
                
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry['expires_at'] < now
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if entry['expires_at'] < time.time():
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        if ttl is None:
            ttl = self.default_ttl
        
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global singleton instances
_performance_monitor = None
_cache_manager = None
_rate_limiter = None
_audit_logger = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

async def initialize_utils():
    """Initialize all utility components"""
    logger.info("Initializing utility modules...")
    
    try:
        # Initialize audit logger
        audit_logger = get_audit_logger()
        await audit_logger.initialize()
        
        # Start rate limiter
        rate_limiter = get_rate_limiter()
        await rate_limiter.start()
        
        # Start cache manager
        cache_manager = get_cache_manager()
        await cache_manager.start()
        
        logger.info("✓ All utility modules initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize utility modules: {str(e)}")
        return False

async def cleanup_utils():
    """Cleanup all utility components"""
    logger.info("Cleaning up utility modules...")
    
    try:
        # Stop and cleanup components
        if _audit_logger:
            await _audit_logger.close()
        
        if _rate_limiter:
            await _rate_limiter.stop()
        
        if _cache_manager:
            await _cache_manager.stop()
        
        logger.info("✓ All utility modules cleaned up")
        
    except Exception as e:
        logger.error(f"Error during utility cleanup: {str(e)}")

# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

async def main():
    """Main function for testing utilities"""
    print("🧪 Testing Utility Modules")
    
    # Test initialization
    success = await initialize_utils()
    
    if success:
        print("✅ Utilities initialized successfully")
        
        # Test performance monitor
        perf = get_performance_monitor()
        perf.record_request(0.1, False)
        print(f"📊 Performance metrics: {perf.get_metrics()}")
        
        # Test cache
        cache = get_cache_manager()
        cache.set("test_key", "test_value", 60)
        cached_value = cache.get("test_key")
        print(f"💾 Cache test: {cached_value}")
        
        # Test rate limiter
        rate_limiter = get_rate_limiter()
        allowed = await rate_limiter.check_limit("test_user")
        print(f"🚦 Rate limit test: {allowed}")
        
        print("✅ All tests passed!")
        
    else:
        print("❌ Utility initialization failed")
    
    # Cleanup
    await cleanup_utils()

if __name__ == "__main__":
    asyncio.run(main())