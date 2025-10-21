// app-controller.js

import AuthManager from './auth.js';
import NiceClassManager from './nice-classes.js';
import SearchManager from './search.js';
import ResultsManager from './results.js';
import QuestionnaireManager from './questionnaire.js';

class TrademarkApp {
    constructor() {
        this.authManager = new AuthManager();
        this.niceClassManager = new NiceClassManager();
        this.questionnaireManager = new QuestionnaireManager(this.niceClassManager); 
        this.searchManager = new SearchManager(this.authManager, this.niceClassManager, this.questionnaireManager);
        this.resultsManager = new ResultsManager(this.authManager, this);
        this.ui = {}; 
        this.searchContext = 'clearance'; 
    }

    initialize() {
        console.log('Initializing Enhanced Trademark Search App');
        
        this.ui = {
            loginPanel: document.getElementById('loginPanel'),
            mainApp: document.getElementById('mainApp'),
            businessContextButton: document.getElementById('businessContextButton'),
            userDisplay: document.getElementById('userDisplay'),
            resultsContainer: document.getElementById('resultsContainer'),
            loginForm: document.getElementById('loginForm'),
            logoutButton: document.getElementById('logoutButton'),
        };

        this.setupEventHandlers();
        this.searchManager.onSearchComplete = (results, mode) => this.resultsManager.displayResults(results, mode);
        
        this.authManager.logout();
        this.showLoginPanel();
    }

    setupEventHandlers() {
        if (this.ui.loginForm) {
            this.ui.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }
        if (this.ui.logoutButton) {
            this.ui.logoutButton.addEventListener('click', () => this.handleLogout());
        }
        
        document.getElementById('knockout-search-context-btn')?.addEventListener('click', () => this.setSearchContextAndFocus('knockout'));
        document.getElementById('clearance-search-context-btn')?.addEventListener('click', () => this.setSearchContextAndFocus('clearance'));
        document.getElementById('metrics-context-btn')?.addEventListener('click', () => this.navigateToMetrics());
        document.getElementById('cease-desist-context-btn')?.addEventListener('click', () => alert('Cease & Desist analysis feature coming soon!'));
        document.getElementById('learn-more-context-btn')?.addEventListener('click', () => alert('Educational resources feature coming soon!'));

    }

    async handleLogin(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const username = formData.get('username');
        const password = formData.get('password');

        try {
            // ANNOTATION: The login method in auth.js already correctly handles the cookie flow.
            await this.authManager.login(username, password);
            await this.showMainApp();
        } catch (error) {
            this.showError(`Login failed: ${error.message}`);
        }
    }

    handleLogout() {
        this.authManager.logout();
        this.showLoginPanel();
    }

    async showMainApp() {
        this.ui.loginPanel.classList.add('hidden');
        this.ui.mainApp.classList.remove('hidden');

        const usernameEl = document.getElementById('username');
        if (this.ui.userDisplay && usernameEl && this.authManager.userInfo) {
            usernameEl.textContent = this.authManager.userInfo.username;
            this.ui.userDisplay.classList.remove('hidden');
        }
        this.niceClassManager.initialize();
        await this.niceClassManager.loadFromAPI();
        this.searchManager.initialize();
        this.questionnaireManager.initialize();
    }

    showLoginPanel() {
        if(this.ui.mainApp) this.ui.mainApp.classList.add('hidden');
        if(this.ui.userDisplay) this.ui.userDisplay.classList.add('hidden');
        if(this.ui.loginPanel) this.ui.loginPanel.classList.remove('hidden');
        
        if (this.ui.resultsContainer) {
            this.ui.resultsContainer.innerHTML = '';
        }
    }

    async performSelectionBasedInvestigation(selectedSerials, questionnaireResponses, searchId) {
        console.log("App Controller: Initiating common law investigation.");

        const selectedMarks = this.resultsManager.currentResults.matches.filter(
            match => selectedSerials.includes(match.serial_number)
        );

        if (selectedMarks.length === 0) {
            this.showError("Could not find selected marks to investigate.");
            return;
        }

        this.resultsManager.showCommonLawLoadingState();

        try {
            const payload = {
                selected_marks: selectedMarks,
                questionnaire_responses: questionnaireResponses,
                search_id: searchId
            };

            // --- START OF FIX ---
            const response = await fetch('/api/common-law/investigate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include', // ANNOTATION: Replaced getAuthHeaders() with this line.
                body: JSON.stringify(payload)
            });
            // --- END OF FIX ---

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Common law investigation failed.');
            }

            const results = await response.json();
            
            this.resultsManager.displayCommonLawResults(results);

        } catch (error) {
            console.error("Common Law Investigation Error:", error);
            this.resultsManager.showCommonLawErrorState(error.message);
        }
    }

    showError(message) {
        console.error('App error:', message);
        alert(message);
    }
    
    setSearchContextAndFocus(contextType) {
        this.searchContext = contextType;
        this.searchManager.setSearchContext(contextType);
        console.log(`Search context set to: ${this.searchContext}`);
        const trademarkInput = document.getElementById('trademark');
        trademarkInput?.focus();
    }
    
    // 10202205
    async navigateToMetrics() {
        console.log("Navigating to Global Metrics view.");
        // Switch to the metrics tab first
        this.resultsManager.activateTab('metricsPanel');

        // Show a loading state
        const globalMetricsContainer = document.getElementById('globalMetricsContainer');
        const metricsContainer = document.getElementById('metricsContainer');
        if (globalMetricsContainer && metricsContainer) {
            metricsContainer.classList.add('hidden');
            globalMetricsContainer.classList.remove('hidden');
            globalMetricsContainer.innerHTML = '<p>Loading global analytics...</p>';
        }

        try {
            const response = await fetch('/api/analytics/global', {
                credentials: 'include'
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to load global analytics data.');
            }
            const data = await response.json();
            // Call the new display function in ResultsManager
            this.resultsManager.displayGlobalMetrics(data);

        } catch (error) {
            console.error(error);
            if (globalMetricsContainer) {
                globalMetricsContainer.innerHTML = `<p class="error" style="color: var(--error-neon);">Could not load global analytics: ${error.message}</p>`;
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new TrademarkApp();
    app.initialize();
});