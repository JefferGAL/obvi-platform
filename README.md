# obvi-platform
# Obvi AI™ (BETA) - Trademark Intelligence Platform 

Welcome to **Obvi AI**, the comprehensive platform designed to streamline and supercharge your trademark search and viability analysis process. Built on a robust, security-first architecture, Obvi AI combines advanced search algorithms, intelligent variation generation, local AI analysis, and professional reporting to empower legal professionals and brand owners.

Our goal is to **defragment IP intelligence**, providing the speed of technology with the depth of a human-centric approach, enabling informed decisions with confidence.

---

## Key Features

* **Advanced Search Capabilities:**
    * **Knockout/Basic Search:** High-speed exact and phonetic matching for quick, direct-hit conflict identification.
    * **Clearance/Enhanced Search:** Deep, comprehensive analysis using fuzzy matching, intelligent variations, and multi-factor similarity scoring (visual, phonetic, conceptual) to uncover nuanced risks.
* **Intelligent Variation Generation:** Our **Word Expansion Engine** generates hundreds of variations across critical categories:
    * Phonetic (Sound-alike)
    * Visual (Look-alike, Leet Speak, character swaps)
    * Morphological (Plurals, forms, compound words)
    * Conceptual (Synonyms, related ideas - using WordNet & custom mappings)
    * *Optional:* AI-driven **Slang & Conceptual** variations via local LLM.
* **NICE Class System & Coordination:**
    * Includes all 45 NICE classes with descriptions.
    * Implements **forced 9↔42 coordination** (Tech Hardware & Software/Services) automatically.
    * Supports **optional coordination** based on USPTO relations.
* **AI-Powered Analysis (Local & Private):** 
    * Integrates with local **Ollama** instances, keeping your search data private.
    * **Questionnaire Processing:** Analyzes user input about business context to recommend optimal NICE classes and search strategies.
    * **Conflict Assessment:** Provides AI-driven summaries and risk stratification for potential conflicts.
    * **Common Law Context:** Analyzes web content from potential common law conflicts in the context of *your* business.
* **Owner-Centric Common Law Investigation:** 
    * Performs targeted web, social media, and business directory searches based on owners identified in the initial USPTO search.
    * Uses AI to verify company identity and summarize web content.
    * Corroborates findings across multiple sources.
* **Performance & Data Integrity:** 
    * Utilizes a **read-only connection** to the primary USPTO trademark database, ensuring data integrity.
    * Employs optimized, multi-step database queries (serial lookup -> enrichment) and class-first filtering.
    * **Real-time progress tracking** via WebSockets (in `enhanced_performance_system.py`).
* **Security-First Architecture:** 
    * **Separate Authentication Database:** Isolates user credentials, sessions, and audit logs from the core trademark data (`separate_auth_db.py`).
    * **Secure Session Management:** Uses `HttpOnly`, `Secure`, `SameSite` cookies instead of `localStorage`.
    * **Role-Based Access Control (RBAC):** Predefined roles (Admin, Analyst, Viewer) with distinct permissions.
    * **Comprehensive Audit Logging:** Tracks authentication events, searches, data access, and security incidents.
    * **Password Security:** Hashing with salt, failed attempt tracking, and account lockout.
* **Professional Reporting & Exports:** 
    * Multiple download formats: **CSV, JSON, TXT, Variations List**.
    * **Advanced Reports:** Generates comprehensive analysis reports in **Markdown (.md)** and **Word (.docx)** (requires Pandoc).
    * Includes multi-perspective analysis and appropriate legal disclaimers.

---

## Architectural Overview

Obvi AI is built on several key principles:

1.  **Security-First Design:** Data isolation (separate auth DB), read-only main DB access, secure session handling, and extensive audit logging are paramount.
2.  **Performance Optimization:** Multi-step queries, class-first filtering, variation caching, and asynchronous operations ensure responsiveness.
3.  **AI-Enhanced Analysis:** Local LLM integration (Ollama) provides sophisticated analysis while maintaining data privacy.
4.  **Professional-Grade Output:** Analysis and reports are designed to meet the standards expected by legal professionals.
5.  **Scalable & Modular:** A clear separation of concerns across different Python modules allows for easier maintenance and future expansion.

---

## Technology Stack

* **Backend:** Python, FastAPI
* **Database:** PostgreSQL (accessed via `asyncpg`)
* **AI/LLM:** Ollama (for local model integration)
* **Similarity/NLP:** Jellyfish, Difflib, NLTK (WordNet - implied by conceptual engine)
* **Authentication:** JWT (via PyJWT), Secure Cookies, PBKDF2 Hashing
* **Frontend:** HTML, CSS, JavaScript (using Chart.js for visualizations)
* **Reporting:** Pypandoc (optional, for `.docx` export)
* **Configuration:** Pydantic (implied by `config_app_config.py`)
* **Data Handling:** Pandas (used in analytics)

---

## File Structure
LATERS

## Getting Started

1.  **Prerequisites:**
    * Python 3.8+
    * PostgreSQL Server
    * Ollama installed and running with a suitable model (e.g., `llama3`, `gemma2`)
    * `pandoc` installed (optional, only for `.docx` report export)

2.  **Clone:** `git clone <repository-url>`
3.  **Environment Setup:**
    * Copy `env_example.sh` to a `.env` file (or configure environment variables directly).
    * **Crucially:** Configure database connection strings for **both** the main USPTO database and the **separate authentication database**. Ensure the main DB user has **read-only** permissions if possible.
    * Set a strong `JWT_SECRET_KEY`.
    * Configure Ollama API URL if not default (`http://localhost:11434`).
    * Configure Google API Key and Search Engine ID if using `common_law_analyzer.py`'s Google Search functionality.

4.  **Database Setup:**
    * Ensure the main USPTO trademark database is populated.
    * The application (via `separate_auth_db.py`) will attempt to **create the separate authentication database** and its tables on first run if `CREATE_AUTH_DB=true` in your environment config. Make sure the database user has creation privileges *or* create the auth database manually first.

5.  **Install Dependencies:** `pip install -r requirements.txt` (Assuming `requirements.txt` exists)
6.  **Run:** `python 5fastapi_tm_main.py`
7.  **Access UI:** Open your browser to `http://localhost:8000` (or the configured port).

---

## Usage

1.  Navigate to `http://localhost:8000`.
2.  Choose a search context (Knockout or Clearance) from the landing page, which redirects you to the main search UI (`/search`).
3.  **Login** using the default credentials (e.g., `admin`/`AdminObvi2025!`) or credentials for users created via `separate_auth_db.py`.
4.  Enter the **Trademark Name** to search.
5.  Select the relevant **NICE Classification(s)** or choose "Search All Classes". Enable/disable optional coordination as needed.
6.  Choose **Search Mode** (Basic or Enhanced). Enhanced mode unlocks variation toggles and threshold sliders.
7.  *(Optional)* Use the **AI Context Interview** button to refine NICE classes and thresholds based on your business description.
8.  Click **Search Trademarks**.
9.  Review results in the **Search Results** tab. Click rows to see detailed score breakdowns.
10. Select relevant results and click **Investigate Selected** to initiate the **Common Law** search (optionally using the Wayfinding chat for context). Review findings in the **Common Law** tab.
11. View per-search analytics in the **Metrics** tab.
12. Click the **Metrics/Analytics** top button to view global USPTO filing trends.
13. Use the **Export** buttons to download data in various formats.

---

## Security Considerations

* **Data Isolation:** User credentials and sessions are stored in a database completely separate from the main trademark data.
* **Read-Only Access:** The primary database connection should be configured as read-only to prevent accidental modification of USPTO data.
* **Secure Sessions:** Authentication relies on secure, `HttpOnly` cookies, mitigating XSS risks associated with `localStorage`.
* **Password Hashing:** Uses PBKDF2 with salt for strong password storage.
* **Audit Trail:** Comprehensive logging of user actions, authentication attempts, and searches is stored in the separate auth database.
* **Local AI:** Ollama integration keeps potentially sensitive trademark search terms within your local environment.
* **Dependencies:** Keep all libraries (FastAPI, PyJWT, asyncpg, etc.) up-to-date.

---

## Contributing

Me Phi ME.

---

## License

LATERS
