// static/js/landing.js
document.addEventListener('DOMContentLoaded', () => {
    const clearanceBtn = document.getElementById('clearance-search-btn');
    const knockoutBtn = document.getElementById('knockout-search-btn');
    const ceaseDesistBtn = document.getElementById('cease-desist-btn');
    const learnMoreBtn = document.getElementById('learn-more-btn');

    if (knockoutBtn) {
        knockoutBtn.addEventListener('click', () => {
            window.location.href = '/search?type=knockout';
        });
    }

    if (clearanceBtn) {
        clearanceBtn.addEventListener('click', () => {
            window.location.href = '/search?type=clearance';
        });
    }

    // Placeholder actions for future features
    if (ceaseDesistBtn) {
        ceaseDesistBtn.addEventListener('click', () => {
            alert('Cease & Desist analysis feature coming soon!');
        });
    }
    
    if (learnMoreBtn) {
         learnMoreBtn.addEventListener('click', () => {
            alert('Educational resources feature coming soon!');
        });
    }
});