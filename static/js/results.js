// results.js

export default class ResultsManager {
    constructor(authManager, appController) {
        this.authManager = authManager;
        this.appController = appController;
        this.currentResults = null;
        this.currentSearchId = null;
        this.currentSearchMode = 'basic';
    }

    displayResults(results, searchMode) {
        this.currentResults = results;
        this.currentSearchMode = searchMode;
        this.currentSearchId = results.search_id;

        const searchResultsPanel = document.getElementById('searchResultsPanel');
        if (searchResultsPanel) {
            if (!results || !results.matches || results.matches.length === 0) {
                this.showEmptyState(searchResultsPanel);
                this.updateDownloadSection(false);
            } else {
                searchResultsPanel.innerHTML = this.generateResultsHTML(results);
                this._addResultEventListeners();
                this.updateDownloadSection(true);
            }
        }

        const metricsPanel = document.getElementById('metricsPanel');
        if (metricsPanel && results.metrics) {
            this.displayMetrics(results.metrics);
        } else if (metricsPanel) {
            metricsPanel.innerHTML = '<p>No metrics data available for this search set.</p>';
        }

        document.getElementById('globalMetricsContainer')?.classList.add('hidden');
        document.getElementById('metricsContainer')?.classList.remove('hidden');

        const commonLawTabButton = document.getElementById('commonLawTabButton');
        if (commonLawTabButton) commonLawTabButton.style.display = 'none';
        
        this.setupTabs();
        this.activateTab('searchResultsPanel');
    }

    displayGlobalMetrics(data) {
        const globalMetricsContainer = document.getElementById('globalMetricsContainer');
        const metricsContainer = document.getElementById('metricsContainer');
        if (!globalMetricsContainer || !metricsContainer) return;

        metricsContainer.classList.add('hidden');
        globalMetricsContainer.classList.remove('hidden');
        globalMetricsContainer.innerHTML = '<h4>Global USPTO Filing Trends</h4>';

        if (data.historical_overview?.applications_by_year) {
            globalMetricsContainer.innerHTML += '<h5>Applications by Year (Since 2000)</h5><canvas id="globalFilingsChart"></canvas>';
            this.renderBarChart('globalFilingsChart', Object.keys(data.historical_overview.applications_by_year).sort(), Object.values(data.historical_overview.applications_by_year), 'Applications');
        }
        if (data.timeline_analysis?.avg_days_to_registration) {
            globalMetricsContainer.innerHTML += '<h5>Average Days to Registration</h5><canvas id="globalProcessingChart"></canvas>';
            this.renderLineChart('globalProcessingChart', Object.keys(data.timeline_analysis.avg_days_to_registration).sort(), Object.values(data.timeline_analysis.avg_days_to_registration), 'Avg. Days');
        }
        if (data.top_filers) {
             globalMetricsContainer.innerHTML += `
                <div class="metrics-grid" style="margin-top: 20px;">
                    <div id="top-orgs-container"></div>
                    <div id="top-firms-container"></div>
                </div>`;
            this.renderTopFilersList('top-orgs-container', 'Top Filing Organizations', data.top_filers.top_organizations);
            this.renderTopFilersList('top-firms-container', 'Top Filing Firms', data.top_filers.top_firms);
        }
    }

    displayMetrics(metricsData) {
        const metricsContainer = document.getElementById('metricsContainer');
        if (!metricsContainer) return;
        
        metricsContainer.innerHTML = '<h4>Result Set Metrics</h4>';
        
        let keyMetricsHtml = '<div class="metrics-grid">';
        keyMetricsHtml += `<div class="metric-card"><strong>Total Results</strong><p class="metric-value">${metricsData.total_matches || 0}</p></div>`;
        keyMetricsHtml += `<div class="metric-card"><strong>High Score</strong><p class="metric-value">${metricsData.high_risk_count || 0}</p></div>`;
        keyMetricsHtml += `<div class="metric-card"><strong>Medium Score</strong><p class="metric-value">${metricsData.medium_risk_count || 0}</p></div>`;
        keyMetricsHtml += `<div class="metric-card"><strong>Avg. Score</strong><p class="metric-value">${((metricsData.average_score || 0) * 100).toFixed(0)}%</p></div>`;
        keyMetricsHtml += '</div>';
        metricsContainer.innerHTML += keyMetricsHtml;

        if (metricsData.filings_by_year && Object.keys(metricsData.filings_by_year).length > 0) {
            metricsContainer.innerHTML += '<h5>Filings by Year (in Result Set)</h5><canvas id="filingsByYearChart"></canvas>';
            this.renderBarChart('filingsByYearChart', Object.keys(metricsData.filings_by_year).sort(), Object.values(metricsData.filings_by_year), 'Applications');
        }
        if (metricsData.status_distribution && Object.keys(metricsData.status_distribution).length > 0) {
            metricsContainer.innerHTML += '<h5 style="margin-top: 20px;">Status Distribution (in Result Set)</h5><canvas id="statusDistChart"></canvas>';
            this.renderPieChart('statusDistChart', Object.keys(metricsData.status_distribution), Object.values(metricsData.status_distribution));
        }
    }

    generateResultsHTML(results) {
        let html = `
            <div class="results-summary">
                <h4>Search Results for "${results.query_trademark}"</h4>
                <div class="summary-stats">
                    <span class="stat">${results.total_matches} matches</span>
                    <span class="stat">${results.execution_time_ms.toFixed(1)}ms</span>
                    <span class="stat">${this.currentSearchMode.toUpperCase()} mode</span>
                </div>
            </div>
            <div class="results-table-container">
                <div class="table-controls">
                    <label class="select-all-container"><input type="checkbox" id="selectAllResults"> Select All</label>
                    <div class="common-law-control">
                        <button id="start-common-law-search" class="btn btn-secondary" disabled>Investigate Selected</button>
                    </div>
                </div>
                <div class="table-wrapper"><table class="results-table"><thead><tr>
                    <th class="checkbox-col">Select</th><th class="mark-col">Trademark</th><th class="serial-col">Serial #</th>
                    <th class="owner-col">Owner</th><th class="status-col">Status</th><th class="classes-col">Classes</th>
                    <th class="risk-col">Risk Level</th>
                </tr></thead><tbody>`;

        results.matches.forEach(match => {
            const classesText = Array.isArray(match.nice_classes) ? match.nice_classes.join(', ') : 'N/A';
            const riskLevel = match.risk_level || 'unknown';
            const scores = match.similarity_scores || {};
            const analysis = match.risk_analysis || {};

            html += `
                <tr class="result-row" data-serial="${match.serial_number}">
                    <td class="checkbox-col"><label class="checkbox-label"><input type="checkbox" class="result-checkbox" value="${match.serial_number}"></label></td>
                    <td class="mark-col">${this.escapeHtml(match.mark_identification)}</td>
                    <td class="serial-col">${match.serial_number}</td>
                    <td class="owner-col">${this.escapeHtml(match.owner || 'N/A')}</td>
                    <td class="status-col"><code>${match.status_code}</code></td>
                    <td class="classes-col">${classesText}</td>
                    <td class="risk-col"><span class="risk-badge risk-${riskLevel.toLowerCase()}">${riskLevel.toUpperCase()}</span></td>
                </tr>
                <tr class="detail-row hidden">
                    <td colspan="7">
                        <div class="detail-content">
                            <h5>Similarity & Risk Breakdown</h5>
                            <div class="detail-section">
                                <strong>Similarity Scores:</strong>
                                <ul>
                                    <li>Phonetic: <span>${(scores.phonetic * 100).toFixed(1)}%</span></li>
                                    <li>Visual: <span>${(scores.visual * 100).toFixed(1)}%</span></li>
                                    <li>Conceptual: <span>${(scores.conceptual * 100).toFixed(1)}%</span></li>
                                    <li><strong>Overall:</strong> <span><strong>${(scores.overall * 100).toFixed(1)}%</strong></span></li>
                                </ul>
                            </div>
                            <div class="detail-section">
                                <strong>Risk Factors:</strong>
                                <ul>
                                    <li>${this.escapeHtml(analysis.status_impact || 'N/A')}</li>
                                    <li>${this.escapeHtml(analysis.threshold_breach || 'No threshold breaches.')}</li>
                                    <li>${this.escapeHtml(analysis.reasoning || 'Standard analysis.')}</li>
                                </ul>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        });

        html += `</tbody></table></div></div>`;
        return html;
    }

    _addResultEventListeners() {
        const commonLawBtn = document.getElementById('start-common-law-search');
        const checkboxes = document.querySelectorAll('#searchResultsPanel .result-checkbox');
        const selectAllCheckbox = document.getElementById('selectAllResults');

        const resultRows = document.querySelectorAll('#searchResultsPanel .result-row');
        resultRows.forEach(row => {
            row.addEventListener('click', (event) => {
                if (event.target.type === 'checkbox' || event.target.tagName === 'LABEL') return;
                const detailRow = row.nextElementSibling;
                if (detailRow && detailRow.classList.contains('detail-row')) {
                    detailRow.classList.toggle('hidden');
                }
            });
        });

        const updateButtonState = () => {
            if (commonLawBtn) {
                const anyChecked = Array.from(checkboxes).some(cb => cb.checked);
                commonLawBtn.disabled = !anyChecked;
            }
        };

        checkboxes.forEach(checkbox => checkbox.addEventListener('change', updateButtonState));
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => {
                checkboxes.forEach(cb => cb.checked = e.target.checked);
                updateButtonState();
            });
        }
        if (commonLawBtn) {
            commonLawBtn.addEventListener('click', () => {
                const selectedSerials = Array.from(document.querySelectorAll('#searchResultsPanel .result-checkbox:checked')).map(cb => cb.value);
                if (selectedSerials.length > 0) this.appController.performSelectionBasedInvestigation(selectedSerials, null, this.currentSearchId);
            });
        }
        
        document.getElementById('downloadCsvBtn')?.addEventListener('click', () => this.downloadResults('csv'));
        document.getElementById('downloadJsonBtn')?.addEventListener('click', () => this.downloadResults('json'));
        document.getElementById('downloadTxtBtn')?.addEventListener('click', () => this.downloadResults('txt'));
        document.getElementById('downloadReportMdBtn')?.addEventListener('click', () => this.downloadResults('report_md'));
        document.getElementById('downloadReportDocxBtn')?.addEventListener('click', () => this.downloadResults('report_docx'));
        updateButtonState();
    }
    
    // NOTE: For brevity, the rest of the functions are collapsed as they are unchanged.
    // Includes: displayCommonLawResults, renderBarChart, renderPieChart, tab functions, filter functions, etc.
    displayCommonLawResults(results) { const commonLawPanel = document.getElementById('commonLawPanel'); const commonLawTabButton = document.getElementById('commonLawTabButton'); if (!commonLawPanel || !commonLawTabButton) return; commonLawPanel.innerHTML = ''; if (!results || !results.success) { this.showCommonLawErrorState(results.detail || 'The investigation returned no data.'); commonLawTabButton.style.display = 'block'; this.activateTab('commonLawPanel'); return; } let html = `<h4>Common Law Investigation Results</h4><p>${this.escapeHtml(results.overall_risk_summary?.summary || 'Analysis complete.')}</p>`; const investigationData = results.investigation_results; if (!investigationData || Object.keys(investigationData).length === 0) { html += '<p>No significant common law findings for the selected marks.</p>'; } else { for (const [mark, owners] of Object.entries(investigationData)) { html += `<div class="common-law-mark-group"><h5>Mark: "${this.escapeHtml(mark)}"</h5>`; for (const [owner, data] of Object.entries(owners)) { html += `<div class="common-law-owner-group"><h6>Owner: ${this.escapeHtml(owner)}</h6>`; const findings = data.common_law_findings || []; if (findings.length > 0) { const foundItems = findings.filter(f => f.status === 'found'); const notFoundItems = findings.filter(f => f.status === 'not_found'); if (foundItems.length > 0) { html += '<strong>Positive Findings:</strong><ul>'; foundItems.forEach(finding => { html += `<li><strong>Source:</strong> ${this.escapeHtml(finding.source_name)}<br><strong>Similarity Score:</strong> ${this.escapeHtml(finding.risk_level)}<br><strong>Summary:</strong> ${this.escapeHtml(finding.finding)}<br>${finding.url ? `<a href="${finding.url}" target="_blank" rel="noopener noreferrer">Visit Source</a>` : ''}</li>`; }); html += '</ul>'; } if (notFoundItems.length > 0) { html += '<strong>Sources Checked (No Conflicts Found):</strong><p style="font-size: 0.9em; color: #666;">'; html += notFoundItems.map(f => `✓ ${this.escapeHtml(f.source_name)}`).join(', '); html += '</p>'; } } else { html += '<p>No specific common law findings for this owner.</p>'; } html += `</div>`; } html += `</div>`; } } commonLawPanel.innerHTML = html; commonLawTabButton.style.display = 'block'; this.activateTab('commonLawPanel'); }
    renderBarChart(canvasId, labels, data, label) { const ctx = document.getElementById(canvasId)?.getContext('2d'); if (!ctx) return; new Chart(ctx, { type: 'bar', data: { labels: labels, datasets: [{ label: label, data: data, backgroundColor: 'rgba(74, 144, 226, 0.6)', borderColor: 'rgba(74, 144, 226, 1)', borderWidth: 1 }] }, options: { responsive: true, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } } }); }
    renderPieChart(canvasId, labels, data) { const ctx = document.getElementById(canvasId)?.getContext('2d'); if (!ctx) return; const backgroundColors = labels.map(label => { if (label.toLowerCase().includes('live') || label.toLowerCase().includes('registered')) return 'rgba(74, 144, 226, 0.7)'; if (label.toLowerCase().includes('pending')) return 'rgba(168, 85, 247, 0.7)'; if (label.toLowerCase().includes('dead') || label.toLowerCase().includes('abandoned') || label.toLowerCase().includes('cancelled')) return 'rgba(255, 0, 255, 0.7)'; return 'rgba(147, 149, 152, 0.7)'; }); new Chart(ctx, { type: 'pie', data: { labels: labels, datasets: [{ data: data, backgroundColor: backgroundColors, hoverOffset: 4 }] }, options: { responsive: true, plugins: { legend: { position: 'top' }, tooltip: { callbacks: { label: function(context) { let label = context.label || ''; if (label) { label += ': '; } if (context.parsed !== null) { label += context.parsed + ' (' + (context.dataset.data.reduce((a, b) => a + b, 0) > 0 ? (context.parsed * 100 / context.dataset.data.reduce((a, b) => a + b, 0)).toFixed(1) : 0) + '%)'; } return label; } } } } } }); }
    setupTabs() { const tabButtons = document.querySelectorAll('.results-tabs .tab-button'); tabButtons.forEach(button => { button.removeEventListener('click', this.handleTabClick); button.addEventListener('click', this.handleTabClick.bind(this)); }); }
    handleTabClick(event) { const targetPanelId = event.target.dataset.tab; this.activateTab(targetPanelId); }
    activateTab(panelId) { const tabButtons = document.querySelectorAll('.results-tabs .tab-button'); const tabPanels = document.querySelectorAll('.tab-panel'); tabButtons.forEach(button => { button.classList.toggle('active', button.dataset.tab === panelId); }); tabPanels.forEach(panel => { panel.classList.toggle('active', panel.id === panelId); panel.classList.toggle('hidden', panel.id !== panelId); }); console.log(`Activated tab: ${panelId}`); }
    _filterTableRows() { const filters = Array.from(document.querySelectorAll('#searchResultsPanel .column-filter')); const rows = document.querySelectorAll('#searchResultsPanel .results-table tbody tr.result-row'); const activeFilters = filters.map(f => ({ columnIndex: parseInt(f.dataset.column, 10), value: f.value.toLowerCase() })).filter(f => f.value !== ''); rows.forEach(row => { let isVisible = true; const cells = row.getElementsByTagName('td'); for (const filter of activeFilters) { const cell = cells[filter.columnIndex]; const cellText = cell ? cell.textContent.toLowerCase() : ''; if (!cellText.includes(filter.value)) { isVisible = false; break; } } row.style.display = isVisible ? '' : 'none'; }); }
    updateDownloadSection(visible) { const downloadSection = document.getElementById('downloadSection'); if (downloadSection) downloadSection.classList.toggle('hidden', !visible); }
    showEmptyState(panelElement) { if (panelElement) panelElement.innerHTML = `<div class="empty-state"><h4>No Matches Found</h4><p>Your search did not return any potential conflicts. Try adjusting terms or mode.</p></div>`; }
    escapeHtml(text) { if (!text) return ''; const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }
    showCommonLawLoadingState() { const container = document.getElementById('commonLawPanel'); if (container) container.innerHTML = '<h4>Performing Common Law Investigation...</h4><p>Analyzing websites and other sources...</p>'; const commonLawTabButton = document.getElementById('commonLawTabButton'); if(commonLawTabButton) commonLawTabButton.style.display = 'block'; this.activateTab('commonLawPanel');}
    showCommonLawErrorState(message) { const container = document.getElementById('commonLawPanel'); if (container) container.innerHTML = `<h4>Investigation Failed</h4><p class="error">${this.escapeHtml(message)}</p>`; const commonLawTabButton = document.getElementById('commonLawTabButton'); if(commonLawTabButton) commonLawTabButton.style.display = 'block'; this.activateTab('commonLawPanel'); }
}