# Quick Installation Guide - Obvi Trademark Search System

## 🚀 Fix for Pydantic Error

The error you encountered is due to Pydantic v2 changes. Here's how to fix it:

### Step 1: Install Required Dependencies

```bash
# Install the fixed Pydantic version and required packages
pip3 install --user pydantic-settings==2.1.0
pip3 install --user pydantic==2.5.0
pip3 install --user fastapi==0.104.1
pip3 install --user uvicorn[standard]==0.24.0
pip3 install --user asyncpg==0.29.0
pip3 install --user PyJWT==2.8.0
pip3 install --user cryptography==41.0.7
```

### Step 2: Replace Configuration File

Replace your `config_app_config.py` with the fixed version provided above. The key changes:

- Updated imports for Pydantic v2 compatibility
- Uses `pydantic-settings` package for `BaseSettings`
- Updated validator syntax for Pydantic v2

### Step 3: Use Fixed Startup Verification

Run the fixed startup verification script:

```bash
python3 startup_verification_fixed.py
```

### Step 4: Create Environment File

Create a `.env` file in your project directory:

```bash
# Database Configuration (update with your settings)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=uspto_trademarks
DB_USERNAME=postgres
DB_PASSWORD=your_password_here

# Auth Database (will be created automatically)
AUTH_DB_NAME=trademark_auth
CREATE_AUTH_DB=true

# Security (auto-generated JWT secret)
JWT_SECRET_KEY=auto_generated_secret_key_here
JWT_EXPIRATION_HOURS=24

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
ENVIRONMENT=development
DEBUG=false
```

### Step 5: Run Verification

```bash
python3 startup_verification_fixed.py
```

### Step 6: Start the System (if verification passes)

```bash
python3 fastapi_tm_main.py
```

## 🔧 What the Fix Does

1. **Pydantic v2 Compatibility**: Updates imports and validator syntax
2. **Automatic Dependency Detection**: Checks for missing packages
3. **Better Error Messages**: Clearer feedback on what needs to be fixed
4. **Safe Configuration**: Maintains read-only mode for your existing database

## 📁 File Structure After Setup

```
your_project/
├── config_app_config.py (updated)
├── startup_verification_fixed.py (new)
├── .env (create this)
├── logs/ (auto-created)
├── data/ (auto-created)
├── temp/ (auto-created)
└── other_project_files...
```

## 🎯 Default Login Credentials

Once the system is running:
- **Admin**: admin / AdminObvi2025!
- **Analyst**: analyst / analyst123!
- **Viewer**: viewer / viewer123!

## 🚦 Quick Test

1. Open: http://127.0.0.1:8000/docs
2. Try the `/health` endpoint
3. Login with admin credentials
4. Test a trademark search

## ⚠️ Important Notes

- Your existing `uspto_trademarks` database will NOT be modified
- A separate `trademark_auth` database will be created for user management
- System runs in read-only mode for trademark data protection
- All authentication and audit data is stored separately

## 🆘 If You Still Have Issues

1. **Check Python version**: Ensure you have Python 3.9+
2. **Install all dependencies**: Run the pip install commands above
3. **Check PostgreSQL**: Ensure your trademark database is accessible
4. **Environment variables**: Make sure .env file has correct database credentials
5. **Run verification**: Always run `startup_verification_fixed.py` first

## 📞 Need Help?

If you're still encountering issues:
1. Check the verification output for specific error messages
2. Ensure all pip packages installed successfully
3. Verify your PostgreSQL connection settings
4. Make sure your existing trademark database exists and is accessible