#!/bin/bash

echo "Setting up Obvi Trademark Search System..."

# Check if Python 3.9+ is available
python3 --version || { echo "❌ Python 3.9+ required but not found. Please install Python 3.9 or higher."; exit 1; }

# Install required Python packages
echo "📦 Installing Python dependencies..."
pip3 install --user pydantic-settings==2.1.0
pip3 install --user pydantic==2.5.0
pip3 install --user fastapi==0.104.1
pip3 install --user uvicorn[standard]==0.24.0
pip3 install --user asyncpg==0.29.0
pip3 install --user PyJWT==2.8.0
pip3 install --user cryptography==41.0.7
pip3 install --user nltk==3.8.1
pip3 install --user jellyfish==0.11.2
pip3 install --user python-multipart==0.0.6

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p temp
mkdir -p auth
mkdir -p cache

# Set up environment file template
echo "⚙️ Creating .env template..."
cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=uspto_trademarks
DB_USERNAME=postgres
DB_PASSWORD=your_password_here

# Auth Database
AUTH_DB_NAME=trademark_auth
CREATE_AUTH_DB=true

# Security Configuration
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_EXPIRATION_HOURS=24

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=false
ENVIRONMENT=development

# Security Settings
RATE_LIMIT_PER_MINUTE=100
FAILED_LOGIN_THRESHOLD=5
LOCKOUT_DURATION_MINUTES=15

# Search Configuration
MAX_SEARCH_RESULTS=1000
ENABLE_PHONETIC_EXPANSION=true
ENABLE_VISUAL_EXPANSION=true
ENABLE_MORPHOLOGICAL_EXPANSION=true
ENABLE_CONCEPTUAL_EXPANSION=true

# Common Law Search Settings
ENABLE_COMMON_LAW_SEARCH=true
COMMON_LAW_RATE_LIMIT_MIN=1.0
COMMON_LAW_RATE_LIMIT_MAX=3.0
COMMON_LAW_CONCURRENT_REQUESTS=3
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your database credentials"
echo "2. Ensure PostgreSQL is running with your trademark database"
echo "3. Run startup verification: python3 startup_verification.py"
echo "4. If verification passes, start the API: python3 fastapi_tm_main.py"
echo ""
echo "🔧 Important: Make sure your existing PostgreSQL database 'uspto_trademarks' is accessible"
echo "   The system will create a separate 'trademark_auth' database for user management"