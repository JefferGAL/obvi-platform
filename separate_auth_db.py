#!/usr/bin/env python3
"""
Separate Authentication Database Manager
Creates/manages auth data in a separate database to protect existing trademark database
"""

import asyncio
import logging
import secrets
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone

import asyncpg
from config_app_config import get_config, AppConfig

logger = logging.getLogger(__name__)

class SeparateAuthDatabase:
    """Manages authentication in a separate database, leaving trademark database untouched"""
    
    def __init__(self):
        self.config = get_config().database
        self._pool = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize separate auth database"""
        try:
            # First, try to create the auth database if it doesn't exist
            await self._ensure_auth_database_exists()
            
            # Connect to the auth database
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.auth_database,
                user=self.config.username,
                password=self.config.password,
                min_size=2,
                max_size=5,
                command_timeout=30
            )
            
            # Create auth tables in the separate database
            await self._create_auth_tables()
            
            self._initialized = True
            logger.info(f"✓ Separate auth database initialized: {self.config.auth_database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize separate auth database: {str(e)}")
            raise
    
    async def _ensure_auth_database_exists(self):
        """Create auth database if it doesn't exist"""
        if not self.config.create_auth_db:
            logger.info("Auth database creation disabled - assuming it exists")
            return
        
        try:
            # Connect to postgres database to create auth database
            admin_conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database='postgres',  # Connect to default postgres database
                user=self.config.username,
                password=self.config.password
            )
            
            # Check if auth database exists
            db_exists = await admin_conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_database WHERE datname = $1
                )
            """, self.config.auth_database)
            
            if not db_exists:
                logger.info(f"Creating separate auth database: {self.config.auth_database}")
                await admin_conn.execute(f'CREATE DATABASE "{self.config.auth_database}"')
                logger.info(f"✓ Auth database created: {self.config.auth_database}")
            else:
                logger.info(f"✓ Auth database already exists: {self.config.auth_database}")
            
            await admin_conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create auth database: {str(e)}")
            raise
    
    async def _create_auth_tables(self):
        """Create authentication tables in separate database"""
        async with self._pool.acquire() as conn:
            # Users table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    email TEXT,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP WITH TIME ZONE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_jti TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    is_active BOOLEAN DEFAULT true,
                    ip_address INET,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                )
            """)
            
            # Audit events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    event_category TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    ip_address INET,
                    user_agent TEXT,
                    details JSONB,
                    session_id TEXT
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_events(user_id, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_category_time ON audit_events(event_category, timestamp)")
            
            logger.info("✓ Auth tables created in separate database")
    
    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
    
    def get_pool(self):
        """Get connection pool for auth operations"""
        return self._pool

# Update the UserStore to use separate database
class UserStoreForSeparateDB:
    """User store using completely separate auth database"""
    
    def __init__(self):
        self.auth_db = SeparateAuthDatabase()
        from auth_manager import PasswordManager
        self.password_manager = PasswordManager()
    
    async def initialize(self):
        """Initialize using separate auth database"""
        await self.auth_db.initialize()
        await self._create_default_users()
        logger.info("✓ User store initialized with separate auth database")
    
    async def _create_default_users(self):
        """Create default users in separate auth database"""
        default_users = [
            {
                'username': 'admin',
                'password': 'AdminObvi2025!',
                'role': 'admin',
                'email': 'admin@obvi.com',
                'full_name': 'System Administrator'
            },
            {
                'username': 'analyst',
                'password': 'analyst123!',
                'role': 'analyst', 
                'email': 'analyst@obvi.com',
                'full_name': 'Trademark Analyst'
            },
            {
                'username': 'viewer',
                'password': 'viewer123!',
                'role': 'viewer',
                'email': 'viewer@obvi.com',
                'full_name': 'Trademark Viewer'
            }
        ]
        
        for user_data in default_users:
            try:
                await self.create_user(**user_data)
            except Exception as e:
                logger.debug(f"Default user creation skipped for {user_data['username']}: {str(e)}")
    
    async def create_user(
        self,
        username: str,
        password: str,
        role: str = 'viewer',
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        permissions: List[str] = None
    ) -> Dict[str, Any]:
        """Create user in separate auth database"""
        
        user_id = secrets.token_urlsafe(16)
        password_hash, password_salt = self.password_manager.hash_password(password)
        #permissions_json = permissions or [] #
        permissions_json = json.dumps(permissions or [])

        async with self.auth_db.get_pool().acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO users (
                        user_id, username, password_hash, password_salt, 
                        role, permissions, email, full_name
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user_id, username, password_hash, password_salt,
                     role, permissions_json, email, full_name)
                
                logger.info(f"User created in separate auth DB: {username} ({role})")
                
                return {
                    'user_id': user_id,
                    'username': username,
                    'role': role,
                    'email': email,
                    'full_name': full_name
                }
                
            except asyncpg.UniqueViolationError:
                from auth_manager import AuthenticationError
                raise AuthenticationError(f"Username '{username}' already exists")
    
    async def _handle_failed_login(self, conn: asyncpg.Connection, user_id: str):
        """Handle failed login attempt"""
        config = get_config().security
        
        # Increment failed attempts
        await conn.execute("""
            UPDATE users 
            SET failed_login_attempts = failed_login_attempts + 1
            WHERE user_id = $1
        """, user_id)
        
        # Check if user should be locked
        failed_attempts = await conn.fetchval("""
            SELECT failed_login_attempts FROM users WHERE user_id = $1
        """, user_id)
        
        if failed_attempts and failed_attempts >= config.failed_login_threshold:
            # FIXED: Use timezone-aware datetime
            locked_until = datetime.now(timezone.utc) + timedelta(minutes=config.lockout_duration_minutes)
            
            await conn.execute("""
                UPDATE users 
                SET locked_until = $1
                WHERE user_id = $2
            """, locked_until, user_id)
            
            logger.warning(f"User {user_id} locked due to {failed_attempts} failed login attempts")

    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user from separate auth database - FIXED"""
        
        async with self.auth_db.get_pool().acquire() as conn:
            user_row = await conn.fetchrow("""
                SELECT * FROM users WHERE username = $1 AND is_active = true
            """, username)
            
            if not user_row:
                from auth_manager import UserNotFoundError
                raise UserNotFoundError(f"User '{username}' not found or inactive")
            
            user = dict(user_row)
            
            # FIXED: Check if user is locked with timezone-aware comparison
            if user['locked_until']:
                locked_until = user['locked_until']
                if locked_until.tzinfo is None:
                    locked_until = locked_until.replace(tzinfo=timezone.utc)
                
                current_time = datetime.now(timezone.utc)
                if current_time < locked_until:
                    from auth_manager import AuthenticationError
                    raise AuthenticationError(f"User account is locked until {locked_until}")
            
            # Verify password
            if not self.password_manager.verify_password(
                password, user['password_hash'], user['password_salt']
            ):
                await self._handle_failed_login(conn, user['user_id'])
                from auth_manager import AuthenticationError
                raise AuthenticationError("Invalid password")
            
            # Reset failed login attempts on successful login
            current_timestamp = datetime.now(timezone.utc)
            await conn.execute("""
                UPDATE users 
                SET failed_login_attempts = 0, last_login = $1, locked_until = NULL
                WHERE user_id = $2
            """, current_timestamp, user['user_id'])
            
            # FIXED: Handle permissions properly
            permissions = user['permissions']
            if isinstance(permissions, str):
                try:
                    permissions = json.loads(permissions)
                except (json.JSONDecodeError, TypeError):
                    permissions = []
            elif permissions is None:
                permissions = []
            
            return {
                'user_id': user['user_id'],
                'username': user['username'],
                'role': user['role'],
                'permissions': permissions,
                'email': user['email'],
                'full_name': user['full_name'],
                'last_login': user['last_login']
            }
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID from separate auth database"""
        async with self.auth_db.get_pool().acquire() as conn:
            user_row = await conn.fetchrow("""
                SELECT * FROM users WHERE user_id = $1 AND is_active = true
            """, user_id)
            
            if user_row:
                return dict(user_row)
            
            return None
    
    async def create_session(
        self,
        user_id: str,
        token_jti: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:

        """Create user session in separate auth database"""
        session_id = secrets.token_urlsafe(24)

        # Ensure expires_at is timezone-aware with UTC 08032025
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        elif expires_at.tzinfo != timezone.utc:
            expires_at = expires_at.astimezone(timezone.utc)
        #####


        async with self.auth_db.get_pool().acquire() as conn:
            await conn.execute("""
                INSERT INTO user_sessions (
                    session_id, user_id, token_jti, expires_at, ip_address, user_agent
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, session_id, user_id, token_jti, expires_at, ip_address, user_agent)
        
        return session_id
    
    async def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        async with self.auth_db.get_pool().acquire() as conn:
            await conn.execute("""
                UPDATE user_sessions SET is_active = false WHERE session_id = $1
            """, session_id)
    
    async def close(self):
        """Close auth database connections"""
        await self.auth_db.close()

# Separate Audit Logger for separate database
class SeparateAuditLogger:
    """Audit logger using separate database to protect trademark database"""
    # 08092025 added ", json.dumps(details, default=str)" to "async with self.auth..." for functions passing dict obj to postgres
    # ...log search event, log data access, log search completion, log error event,  and log security event
    
    def __init__(self):
        self.auth_db = SeparateAuthDatabase()
        self._initialized = False
    
    async def initialize(self):
        """Initialize audit logging in separate database"""
        await self.auth_db.initialize()
        self._initialized = True
        logger.info("✓ Audit logging initialized in separate database")
    
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
        if not self._initialized:
            return
        
        try:
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, ip_address, 
                        user_agent, details, session_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user_id, event_type, 'auth', success, ip_address, 
                    user_agent, json.dumps(details) if details else None, session_id)
        except Exception as e:
            logger.error(f"Failed to log auth event: {str(e)}")
    
    async def log_data_access(
        self,
        user_id: str,
        operation: str,
        table_name: Optional[str] = None,
        record_count: Optional[int] = None,
        query: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        ip_address: Optional[str] = None,
        success: bool = True
    ):
        """Log data access event"""
        if not self._initialized:
            return
        
        try:
            # Hash the query for privacy
            query_hash = None
            if query:
                import hashlib
                query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            
            details = {
                'table_name': table_name,
                'record_count': record_count,
                'query_hash': query_hash,
                'execution_time_ms': execution_time_ms,
                'success': success
            }
            
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, ip_address, details
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, operation, 'data_access', success, ip_address, json.dumps(details, default=str))
        except Exception as e:
            logger.error(f"Failed to log data access: {str(e)}")
    
    async def log_search_event(
        self,
        user_id: str,
        search_type: str,
        query: str,
        search_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        result_count: Optional[int] = None,
        execution_time_ms: Optional[float] = None,
        ip_address: Optional[str] = None
    ):
        """Log search event"""
        if not self._initialized:
            return
        
        try:
            details = {
                'search_id': search_id,
                'query': query,
                'filters': filters,
                'result_count': result_count,
                'execution_time_ms': execution_time_ms
            }
            
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, ip_address, details
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, search_type, 'search', True, ip_address, json.dumps(details, default=str))
        except Exception as e:
            logger.error(f"Failed to log search event: {str(e)}")
    
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict] = None,
        action_taken: Optional[str] = None
    ):
        """Log security event"""
        if not self._initialized:
            return
        
        try:
            event_details = details or {}
            event_details.update({
                'severity': severity,
                'description': description,
                'action_taken': action_taken
            })
            
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, ip_address,
                        user_agent, details
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, user_id, event_type, 'security', False, ip_address,
                    user_agent, json.dumps(event_details, default=str))
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
    
    async def log_error_event(
        self,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ):
        """Log error event"""
        if not self._initialized:
            return
        
        try:
            details = {
                'error_message': error_message,
                'stack_trace': stack_trace,
                'context': context
            }
            
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, ip_address, details
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, user_id, error_type, 'error', False, ip_address, json.dumps(details, default=str))
        except Exception as e:
            logger.error(f"Failed to log error event: {str(e)}")
    
    async def log_search_completion(
        self,
        user_id: str,
        search_id: str,
        result_count: int,
        execution_time_ms: float
    ):
        """Log search completion"""
        if not self._initialized:
            return
        
        try:
            details = {
                'search_id': search_id,
                'result_count': result_count,
                'execution_time_ms': execution_time_ms,
                'status': 'completed'
            }
            
            async with self.auth_db.get_pool().acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events (
                        user_id, event_type, event_category, success, details
                    ) VALUES ($1, $2, $3, $4, $5)
                """, user_id, 'search_completion', 'search', True, json.dumps(details, default=str))
        except Exception as e:
            logger.error(f"Failed to log search completion: {str(e)}")
    
    async def get_user_activity_summary(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user activity summary"""
        if not self._initialized:
            return {}
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with self.auth_db.get_pool().acquire() as conn:
                # Get event counts by category
                events = await conn.fetch("""
                    SELECT event_category, event_type, COUNT(*) as count
                    FROM audit_events
                    WHERE user_id = $1 AND timestamp > $2
                    GROUP BY event_category, event_type
                    ORDER BY event_category, event_type
                """, user_id, cutoff_date)
                
                # Get search statistics
                search_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as search_count,
                        AVG((details->>'execution_time_ms')::float) as avg_time
                    FROM audit_events
                    WHERE user_id = $1 AND event_category = 'search' AND timestamp > $2
                """, user_id, cutoff_date)
                
                activity_summary = {
                    'user_id': user_id,
                    'period_days': days,
                    'events_by_category': {},
                    'search_count': search_stats['search_count'] if search_stats else 0,
                    'avg_search_time_ms': search_stats['avg_time'] if search_stats else 0
                }
                
                # Organize events by category
                for event in events:
                    category = event['event_category']
                    if category not in activity_summary['events_by_category']:
                        activity_summary['events_by_category'][category] = {}
                    activity_summary['events_by_category'][category][event['event_type']] = event['count']
                
                return activity_summary
                
        except Exception as e:
            logger.error(f"Failed to get user activity summary: {str(e)}")
            return {}
    
    async def get_security_alerts(self, severity: str = "high", days: int = 7) -> List[Dict]:
        """Get recent security alerts"""
        if not self._initialized:
            return []
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with self.auth_db.get_pool().acquire() as conn:
                alerts = await conn.fetch("""
                    SELECT 
                        timestamp, user_id, event_type, ip_address, 
                        user_agent, details
                    FROM audit_events
                    WHERE event_category = 'security' 
                    AND details->>'severity' = $1 
                    AND timestamp > $2
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, severity, cutoff_date)
                
                return [dict(alert) for alert in alerts]
                
        except Exception as e:
            logger.error(f"Failed to get security alerts: {str(e)}")
            return []
    
    async def close(self):
        """Close audit logger"""
        await self.auth_db.close()
    