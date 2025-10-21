// search.js

export default class SearchManager {
    constructor(authManager, niceClassManager, questionnaireManager) {
        this.authManager = authManager;
        this.niceClassManager = niceClassManager;
        this.questionnaireManager = questionnaireManager;
        this.currentSearchMode = 'basic';
        this.searchContext = 'clearance';
        this.isSearching = false;
        this.onSearchComplete = null;
    }

    initialize() {
        if (this.isInitialized) {
            return;
        }
        this.initializeSearchForm();
        this.initializeModeSlider();
        this.initializeAllClassesToggle();
        this.initializeAdvancedOptions();
        this.isInitialized = true;
    }

    initializeSearchForm() {
        const form = document.getElementById('searchForm');
        form?.addEventListener('submit', (e) => this.handleSearchSubmit(e));
    }

    initializeModeSlider() {
        const slider = document.getElementById('searchModeSlider');
        const descBasic = document.getElementById('modeDescBasic');
        const descEnhanced = document.getElementById('modeDescEnhanced');
        const enhancedOptions = document.getElementById('enhancedModeOptions');
        const questionnaireBtn = document.getElementById('startQuestionnaireBtn');

        if (!slider || !descBasic || !descEnhanced || !enhancedOptions || !questionnaireBtn) return;

        questionnaireBtn.addEventListener('click', () => {
            if (this.questionnaireManager) {
                this.questionnaireManager.open();
            }
        });

        const updateSliderState = () => {
            if (slider.value === '0') {
                this.currentSearchMode = 'basic';
                descBasic.classList.add('active');
                descEnhanced.classList.remove('active');
                enhancedOptions.classList.add('hidden');
                questionnaireBtn.disabled = true;
            } else {
                this.currentSearchMode = 'enhanced';
                descBasic.classList.remove('active');
                descEnhanced.classList.add('active');
                enhancedOptions.classList.remove('hidden');
                questionnaireBtn.disabled = false;
            }
        };
        slider.addEventListener('input', updateSliderState);
        updateSliderState();
    }

    setSearchContext(context) {
        this.searchContext = context;
        const contextDisplay = document.getElementById('searchContextDisplay');
        if (contextDisplay) {
            contextDisplay.textContent = `Mode: ${context.charAt(0).toUpperCase() + context.slice(1)}`;
        }
    }

    initializeAllClassesToggle() {
        const toggle = document.getElementById('all-classes-toggle');
        const grid = document.getElementById('niceClassesGrid');
        if (!toggle || !grid) return;

        toggle.addEventListener('change', () => {
            grid.classList.toggle('disabled', toggle.checked);
            if (toggle.checked) {
                this.niceClassManager.deselectAll();
            }
        });
    }
    
    initializeAdvancedOptions() {
        const toggle = document.getElementById('advancedOptionsToggle');
        const content = document.getElementById('advancedOptionsContent');
        if (!toggle || !content) return;

        toggle.addEventListener('click', () => {
            content.classList.toggle('hidden');
            toggle.querySelector('.arrow').classList.toggle('down');
        });
    }

    async handleSearchSubmit(event) {
        event.preventDefault();
        if (this.isSearching) return;
        
        const trademark = document.getElementById('trademark').value.trim();
        if (!trademark) {
            this.showError('Please enter a mark to analyze');
            return;
        }

        if (trademark.includes('|')) {
            document.getElementById('useVariationsToggle').checked = false;
            document.getElementById('slangSearchToggle').checked = false;
        }

        const allClassestoggle = document.getElementById('all-classes-toggle');
        let selectedClasses = [];

        if (allClassestoggle && allClassestoggle.checked) {
            selectedClasses = ['all_classes'];
        } else {
            selectedClasses = this.niceClassManager.getSelectedClasses();
            if (selectedClasses.length === 0) {
                this.showError('Please select at least one NICE class or enable "Search All NICE Classes"');
                return;
            }
        }

        this.setSearchingState(true);
        try {
            const searchData = {
                trademark: trademark,
                classes: selectedClasses,
                search_mode: this.currentSearchMode,
                search_context: this.searchContext,
                enable_slang_search: document.getElementById('slangSearchToggle').checked,
                use_variations: document.getElementById('useVariationsToggle').checked,
                phonetic_threshold: parseFloat(document.getElementById('phoneticThreshold').value) / 100,
                visual_threshold: parseFloat(document.getElementById('visualThreshold').value) / 100,
                conceptual_threshold: parseFloat(document.getElementById('conceptualThreshold').value) / 100,
                max_results: parseInt(document.getElementById('maxResults').value)
            };

            // --- START OF FIX ---
            const response = await fetch('/search/trademark', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include', // ANNOTATION: Replaced getAuthHeaders() with this line.
                body: JSON.stringify(searchData)
            });
            // --- END OF FIX ---

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Search failed: ${response.status}`);
            }
            const results = await response.json();
            if (this.onSearchComplete) {
                this.onSearchComplete(results, this.currentSearchMode);
            }
        } catch (error) {
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.setSearchingState(false);
        }
    }

    setSearchingState(searching) {
        const button = document.getElementById('searchButton');
        const buttonText = document.getElementById('searchButtonText'); 
        if (button && buttonText) {
            button.disabled = searching;
            buttonText.textContent = searching ? 'Searching...' : 'Search Trademarks';
        }
    }

    showError(message) {
        console.error('Search error:', message);
        alert(message);
    }
}