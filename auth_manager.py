#!/usr/bin/env python3
"""
Authentication Manager Module - COMPLETELY FIXED
Secure JWT-based authentication and user management with proper timezone handling
"""

import asyncio
import logging
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import os

# JWT and cryptography
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# PostgreSQL async database
import asyncpg
try:
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Fixed imports
from config_app_config import get_config

# fastapi
# ANNOTATION: Cookie is now imported, and OAuth2PasswordBearer is removed.
from fastapi import FastAPI, HTTPException, Depends, Query, Security, BackgroundTasks, status, Cookie
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Authentication related errors"""
    pass

class TokenError(AuthenticationError):
    """JWT token related errors"""
    pass

class UserNotFoundError(AuthenticationError):
    """User not found error"""
    pass

class PasswordManager:
    """Secure password hashing and verification"""
    
    def __init__(self):
        self.salt_length = 32
        self.iterations = 100000  # PBKDF2 iterations
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_bytes(self.salt_length)
        
        # Use PBKDF2 for secure password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
        )
        
        password_hash = kdf.derive(password.encode('utf-8'))
        
        # Return hex-encoded hash and salt
        return password_hash.hex(), salt.hex()
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt = bytes.fromhex(stored_salt)
            expected_hash, _ = self.hash_password(password, salt)
            
            # Constant-time comparison to prevent timing attacks
            return secrets.compare_digest(expected_hash, stored_hash)
            
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False

class TokenManager:
    """JWT token management with proper timezone handling"""
    
    def __init__(self, config):
        self.config = config
        self.secret_key = config.jwt_secret_key
        self.algorithm = config.jwt_algorithm
        self.expiration_hours = config.jwt_expiration_hours
        
        if not JWT_AVAILABLE:
            raise AuthenticationError("JWT library not available")
    
    def create_access_token(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create JWT access token with proper timezone handling"""
        # Use UTC timezone consistently
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            'user_id': user_data['user_id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'permissions': user_data.get('permissions', []),
            'exp': int(expiry.timestamp()),  # Convert to integer timestamp
            'jti': secrets.token_urlsafe(16)
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            return {
                'access_token': token,
                'token_type': 'bearer',
                'expires_in': int((expiry - now).total_seconds()),
                'expires_at': int(expiry.timestamp())  # Return as integer timestamp
            }
            
        except Exception as e:
            logger.error(f"Token creation error: {str(e)}")
            raise TokenError(f"Failed to create token: {str(e)}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            # Handle potential encoding issues
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            
            token = token.strip()  # Remove any whitespace
            
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={'verify_exp': True}
            )
            
            # Additional validation
            required_fields = ['user_id', 'username', 'role', 'exp']
            for field in required_fields:
                if field not in payload:
                    raise TokenError(f"Missing required field: {field}")
            
            # Check if token is expired
            if payload['exp'] < time.time():
                raise TokenError("Token has expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise TokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise TokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise TokenError(f"Token verification failed: {str(e)}")
    
    def refresh_token(self, token: str) -> Dict[str, Any]:
        """Refresh an existing token"""
        try:
            # Verify current token (allowing expired tokens for refresh)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={'verify_exp': False}  # Allow expired for refresh
            )
            
            # Check if token is too old to refresh (e.g., more than 7 days expired)
            max_refresh_age = 7 * 24 * 3600  # 7 days in seconds
            if time.time() - payload['exp'] > max_refresh_age:
                raise TokenError("Token too old to refresh")
            
            # Create new token with same user data
            user_data = {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'role': payload['role'],
                'permissions': payload.get('permissions', [])
            }
            
            return self.create_access_token(user_data)
            
        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            raise TokenError(f"Token refresh failed: {str(e)}")


class AuthManager:
    """Main authentication manager with proper timezone handling - FIXED"""
    
    def __init__(self):
        self.config = get_config()
        self.user_store = None  # Will be set externally
        self.token_manager = TokenManager(self.config.security)
        self.password_manager = PasswordManager()
        self._initialized = False
    
    async def initialize(self):
        """Initialize authentication system - FIXED"""
        if self._initialized:
            logger.info("✓ Authentication manager already initialized")
            return
        
        try:
            # Check if user_store is properly set
            if self.user_store is None:
                logger.error("❌ User store not set! This should be set externally.")
                raise AuthenticationError("User store not configured")
            
            # Initialize the user store
            if hasattr(self.user_store, 'initialize'):
                await self.user_store.initialize()
                logger.info("✓ User store initialized")
            
            self._initialized = True
            logger.info("✓ Authentication manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Authentication manager initialization failed: {str(e)}")
            raise AuthenticationError(f"Failed to initialize auth manager: {str(e)}")
    
    async def close(self):
        """Cleanup authentication system"""
        if hasattr(self, 'user_store') and self.user_store:
            if hasattr(self.user_store, 'close'):
                await self.user_store.close()
        logger.info("✓ Authentication manager closed")
    
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and return result"""
        if not self._initialized:
            logger.error("❌ Authentication manager not initialized")
            raise AuthenticationError("Authentication manager not initialized")
        
        if not self.user_store:
            logger.error("❌ User store not available")
            raise AuthenticationError("User store not available")
        
        try:
            logger.debug(f"🔐 Attempting to authenticate user: {username}")
            
            # Authenticate with user store
            user_data = await self.user_store.authenticate_user(username, password)
            
            logger.info(f"✅ User authenticated successfully: {username}")
            return {
                'success': True,
                'user': user_data,
                'message': 'Authentication successful'
            }
            
        except (UserNotFoundError, AuthenticationError) as e:
            logger.warning(f"❌ Authentication failed for {username}: {str(e)}")
            return {
                'success': False,
                'user': None,
                'message': str(e)
            }
        except Exception as e:
            logger.error(f"❌ Authentication system error for {username}: {str(e)}")
            return {
                'success': False,
                'user': None,
                'message': 'Authentication system error'
            }
    
    # ANNOTATION: This is the new function to log security events. It is placed here as a
    # method of the AuthManager class, consistent with the class's responsibilities.
    # Using async - might log to DB - obvi a different one!!!
    async def log_security_event(self, username: str, event_type: str, details: str):
        """Logs a security-related event for a user."""
        try:
            security_logger = logging.getLogger("security_events")
            security_logger.warning(f"User: '{username}', Event: '{event_type}', Details: '{details}'")
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    async def create_access_token(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create access token for authenticated user with proper timezone handling"""
        try:
            logger.debug(f"🎫 Creating access token for user: {user_data.get('username')}")
            
            token_data = self.token_manager.create_access_token(user_data)
            
            # Convert timestamp back to timezone-aware datetime for session creation
            expires_at = datetime.fromtimestamp(token_data['expires_at'], tz=timezone.utc)
            
            # Get token payload to extract jti
            payload = jwt.decode(
                token_data['access_token'],
                self.config.security.jwt_secret_key,
                algorithms=[self.config.security.jwt_algorithm],
                options={'verify_exp': False}
            )
            
            # Create session record if user_store supports it
            session_id = None
            if hasattr(self.user_store, 'create_session'):
                try:
                    session_id = await self.user_store.create_session(
                        user_id=user_data['user_id'],
                        token_jti=payload['jti'],
                        expires_at=expires_at
                    )
                    logger.debug(f"✓ Session created: {session_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create session: {str(e)}")
            
            if session_id:
                token_data['session_id'] = session_id
            
            logger.info(f"✅ Access token created successfully for: {user_data.get('username')}")
            return token_data
            
        except Exception as e:
            logger.error(f"❌ Token creation error: {str(e)}")
            raise TokenError(f"Failed to create access token: {str(e)}")
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return user information"""
        try:
            logger.debug("🔍 Verifying token...")
            
            # Verify token signature and expiration
            payload = self.token_manager.verify_token(token)
            
            # Get current user data if user_store is available
            user_data = None
            if self.user_store and hasattr(self.user_store, 'get_user_by_id'):
                try:
                    user_data = await self.user_store.get_user_by_id(payload['user_id'])
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch user data: {str(e)}")
            
            if not user_data:
                # Fallback to token payload data
                user_data = {
                    'user_id': payload['user_id'],
                    'username': payload['username'],
                    'role': payload['role'],
                    'permissions': payload.get('permissions', [])
                }
                logger.debug("Using token payload data as fallback")
            
            result = {
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'role': user_data['role'],
                'permissions': user_data.get('permissions', []),
                'token_jti': payload['jti']
            }
            
            logger.debug(f"✅ Token verified for user: {user_data['username']}")
            return result
            
        except TokenError as e:
            logger.warning(f"❌ Token verification failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Token verification error: {str(e)}")
            return None
    
    async def refresh_token(self, token: str) -> Dict[str, Any]:
        """Refresh an existing token"""
        try:
            logger.debug("🔄 Refreshing token...")
            refreshed = self.token_manager.refresh_token(token)
            logger.debug("✅ Token refreshed successfully")
            return refreshed
        except TokenError as e:
            logger.warning(f"❌ Token refresh failed: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"❌ Token refresh error: {str(e)}")
            raise TokenError(f"Token refresh failed: {str(e)}")
    
    # added 08182025 - to clean up temporary data
    async def logout_user(self, session_id: str):
        """Logout user by invalidating session and cleaning up temporary data"""
        try:
            # Invalidate session in database
            if self.user_store and hasattr(self.user_store, 'invalidate_session'):
                await self.user_store.invalidate_session(session_id)
                logger.info(f"✅ User session {session_id} invalidated")
            else:
                logger.warning("⚠️ Session invalidation not supported by user store")
            
            # Clean up common law temporary data
            try:
                from common_law_integration import get_session_storage
                session_storage = get_session_storage()
                cleaned_files = await session_storage.cleanup_session_data(session_id)
                if cleaned_files > 0:
                    logger.info(f"✅ Cleaned up {cleaned_files} temporary common law files for session {session_id}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup temporary data for session {session_id}: {str(cleanup_error)}")
                
        except Exception as e:
            logger.error(f"❌ Logout error: {str(e)}")

    async def create_user(
        self,
        username: str,
        password: str,
        role: str = 'viewer',
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        permissions: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new user"""
        if not self._initialized:
            raise AuthenticationError("Authentication manager not initialized")
        
        if not self.user_store:
            raise AuthenticationError("User store not available")
        
        try:
            logger.debug(f"👤 Creating user: {username} ({role})")
            
            if not hasattr(self.user_store, 'create_user'):
                raise AuthenticationError("User creation not supported by user store")
            
            result = await self.user_store.create_user(
                username=username,
                password=password,
                role=role,
                email=email,
                full_name=full_name,
                permissions=permissions
            )
            
            logger.info(f"✅ User created successfully: {username}")
            return result
            
        except Exception as e:
            logger.error(f"❌ User creation error: {str(e)}")
            raise
    
    def get_user_permissions(self, role: str) -> List[str]:
        """Get default permissions for a role"""
        role_permissions = {
            'admin': [
                'user:create', 'user:read', 'user:update', 'user:delete',
                'trademark:search', 'trademark:analyze', 'trademark:export',
                'system:configure', 'system:monitor'
            ],
            'analyst': [
                'trademark:search', 'trademark:analyze', 'trademark:export',
                'user:read'
            ],
            'viewer': [
                'trademark:search', 'trademark:view'
            ]
        }
        
        return role_permissions.get(role, [])

# Singleton instance
_auth_manager_instance: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    """Get singleton authentication manager instance - FIXED"""
    global _auth_manager_instance
    
    if _auth_manager_instance is None:
        _auth_manager_instance = AuthManager()
        logger.info("🔧 Authentication manager instance created")
    
    return _auth_manager_instance

# ANNOTATION: This is the new dependency function. It extracts the token
# from the 'session_token' cookie instead of the Authorization header.
async def get_current_user_info(session_token: Optional[str] = Cookie(None)) -> Dict[str, Any]:
    """FastAPI dependency to verify a token from a cookie and return the user's information."""
    if session_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated: session token is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_manager = get_auth_manager()
    if not auth_manager:
        raise HTTPException(status_code=503, detail="Authentication service not available.")
    
    user_info = await auth_manager.verify_token(session_token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_info