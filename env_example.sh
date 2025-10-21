# FastAPI Trademark Search System - Environment Configuration
# Copy this file to .env and update with your values

# Application Settings
APP_NAME="Obvi Trademark Search API"
APP_VERSION="1.0.0"
ENVIRONMENT="development"  # development, testing, staging, production
DEBUG=true

# API Server Settings
API_HOST="127.0.0.1"
API_PORT=8000
API_WORKERS=1

# Database Configuration (EXISTING USPTO trademark database)
DB_HOST="localhost"
DB_PORT=5432
DB_NAME="uspto_trademarks"  # YOUR EXISTING DATABASE - WILL NOT BE MODIFIED
DB_USERNAME="postgres"
DB_PASSWORD=""

# Database Safety Settings
DB_READ_ONLY_MODE=true  # CRITICAL: Protects your existing database
ALLOW_SCHEMA_CREATION=false  # CRITICAL: Prevents modification of existing database

# Separate Authentication Database (will be created if needed)
AUTH_DB_NAME="trademark_auth"  # Separate database for auth/audit data
CREATE_AUTH_DB=true  # Allow creation of separate auth database

# Database Connection Pool
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=20
DB_CONNECTION_TIMEOUT=30
DB_QUERY_TIMEOUT=60
DB_MAX_RETRIES=3

# Security Configuration
JWT_SECRET_KEY="your-super-secret-jwt-key-here-make-it-long-and-random-12"
JWT_ALGORITHM="HS256"
JWT_EXPIRATION_HOURS=24

# Session Management
SESSION_TIMEOUT_MINUTES=30
MAX_CONCURRENT_SESSIONS=3

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=150

# Security Thresholds
FAILED_LOGIN_THRESHOLD=5
LOCKOUT_DURATION_MINUTES=15

# CORS Settings
ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"

# Search Configuration
MAX_SEARCH_RESULTS=1000
DEFAULT_BATCH_SIZE=1000

# Feature Flags
ENABLE_PHONETIC_EXPANSION=true
ENABLE_VISUAL_EXPANSION=true
ENABLE_MORPHOLOGICAL_EXPANSION=true
ENABLE_CONCEPTUAL_EXPANSION=true

# AI/LLM Settings (optional)
ENABLE_AI_ANALYSIS=false
AI_MODEL_NAME="gemma3"
AI_API_TIMEOUT=30

# Cache Settings
ENABLE_SEARCH_CACHE=true
CACHE_TTL_MINUTES=60

# Logging Configuration
LOG_LEVEL="INFO"
DB_LOG_LEVEL="INFO"
LOG_DIRECTORY="./logs"
LOG_FILE_MAX_SIZE="10MB"
LOG_FILE_BACKUP_COUNT=5

# Audit Logging
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=90

# Performance Monitoring
LOG_SLOW_QUERIES=true
SLOW_QUERY_THRESHOLD_MS=1000

# File Management
DATA_DIRECTORY="./data"
TEMP_DIRECTORY="./temp"

# Performance Settings
MAX_REQUEST_SIZE=10485760  # 10MB in bytes
REQUEST_TIMEOUT=60

# Feature Toggles
ENABLE_DETAILED_LOGGING=true
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true