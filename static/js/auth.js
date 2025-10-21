// static/js/auth.js

/**
 * Manages client-side authentication state and API calls for login/logout.
 * This version uses a cookie-based session and does not store JWTs in localStorage.
 */
export class AuthManager {
    constructor() {
        this.userInfo = null;
        this.loadUserFromStorage();
    }

    /**
     * Loads user information from localStorage to persist UI state across page loads.
     * The session itself is managed by the browser cookie.
     */
    loadUserFromStorage() {
        const storedUser = localStorage.getItem('userInfo');
        if (storedUser) {
            this.userInfo = JSON.parse(storedUser);
        }
    }

    /**
     * Attempts to log in the user by calling the /auth/login endpoint.
     * The server will set an HTTP-only cookie upon success.
     * @param {string} username - The user's username.
     * @param {string} password - The user's password.
     * @returns {Promise<object>} - A promise that resolves with user info on success.
     */
    async login(username, password) {
        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // ANNOTATION: 'include' tells the browser to send cookies with the request.
            credentials: 'include',
            body: JSON.stringify({ username, password }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Login failed');
        }

        const data = await response.json();
        this.userInfo = data.userInfo;

        // ANNOTATION: Only user info (for display) is stored, NOT the token.
        localStorage.setItem('userInfo', JSON.stringify(this.userInfo));

        return this.userInfo;
    }

    /**
     * Logs out the user by calling the /auth/logout endpoint, which clears the cookie.
     */
    async logout() {
        try {
            await fetch('/auth/logout', {
                method: 'POST',
                // ANNOTATION: Credentials must be included for the server to recognize the session to end.
                credentials: 'include',
            });
        } catch (error) {
            console.error("Logout request failed, clearing local data anyway:", error);
        } finally {
            this.userInfo = null;
            // ANNOTATION: Clear all local session-related data.
            localStorage.removeItem('userInfo');
            // You might want to clear other cached data here as well.
            // sessionStorage.clear();
        }
    }

    /**
     * Checks if there is user info stored locally.
     * This is a UI check, not a real authentication check.
     * @returns {boolean} - True if user info is present.
     */
    isAuthenticated() {
        return this.userInfo !== null;
    }

    /**
     * Gets the currently stored user info.
     * @returns {object|null} - The user info object or null.
     */
    getUserInfo() {
        return this.userInfo;
    }

    // ANNOTATION: The getAuthHeaders() method has been completely removed
    // as it is no longer needed. All fetch calls will use `credentials: 'include'`.
}