sap.ui.define([
    "sap/ui/core/Core"
], function (Core) {
    "use strict";

    /**
     * Theme Service for Dark Mode Support
     * Handles theme switching, persistence, and event notifications
     */
    var ThemeService = {

        // Theme constants
        THEMES: {
            LIGHT: "sap_horizon",
            DARK: "sap_horizon_dark"
        },

        // LocalStorage key
        STORAGE_KEY: "nucleus_dashboard_theme",

        // Event callbacks for theme change notifications
        _callbacks: [],

        /**
         * Initialize the theme service and apply saved preference
         */
        init: function () {
            var sSavedTheme = this._getStoredTheme();
            if (sSavedTheme) {
                this._applyTheme(sSavedTheme);
            }
            return this;
        },

        /**
         * Get the current active theme
         * @returns {string} Current theme name
         */
        getCurrentTheme: function () {
            return Core.getConfiguration().getTheme();
        },

        /**
         * Check if dark mode is currently active
         * @returns {boolean} True if dark theme is active
         */
        isDarkMode: function () {
            return this.getCurrentTheme() === this.THEMES.DARK;
        },

        /**
         * Set a specific theme
         * @param {string} sThemeName - Theme name to apply (sap_horizon or sap_horizon_dark)
         */
        setTheme: function (sThemeName) {
            // Validate theme name
            if (sThemeName !== this.THEMES.LIGHT && sThemeName !== this.THEMES.DARK) {
                console.warn("ThemeService: Invalid theme name:", sThemeName);
                return;
            }

            this._applyTheme(sThemeName);
            this._saveTheme(sThemeName);
            this._fireThemeChanged(sThemeName);
        },

        /**
         * Toggle between light and dark themes
         * @returns {string} The new active theme name
         */
        toggleTheme: function () {
            var sCurrentTheme = this.getCurrentTheme();
            var sNewTheme = sCurrentTheme === this.THEMES.DARK 
                ? this.THEMES.LIGHT 
                : this.THEMES.DARK;

            this.setTheme(sNewTheme);
            return sNewTheme;
        },

        /**
         * Register a callback for theme change events
         * @param {function} fnCallback - Callback function(sNewTheme, sOldTheme)
         * @returns {object} This instance for chaining
         */
        attachThemeChanged: function (fnCallback) {
            if (typeof fnCallback === "function") {
                this._callbacks.push(fnCallback);
            }
            return this;
        },

        /**
         * Unregister a callback for theme change events
         * @param {function} fnCallback - Previously registered callback
         * @returns {object} This instance for chaining
         */
        detachThemeChanged: function (fnCallback) {
            var iIndex = this._callbacks.indexOf(fnCallback);
            if (iIndex > -1) {
                this._callbacks.splice(iIndex, 1);
            }
            return this;
        },

        /**
         * Apply theme using UI5 Core
         * @private
         */
        _applyTheme: function (sThemeName) {
            Core.applyTheme(sThemeName);
        },

        /**
         * Get stored theme from localStorage
         * @private
         * @returns {string|null} Stored theme or null
         */
        _getStoredTheme: function () {
            try {
                return window.localStorage.getItem(this.STORAGE_KEY);
            } catch (e) {
                console.warn("ThemeService: Could not read from localStorage:", e);
                return null;
            }
        },

        /**
         * Save theme preference to localStorage
         * @private
         */
        _saveTheme: function (sThemeName) {
            try {
                window.localStorage.setItem(this.STORAGE_KEY, sThemeName);
            } catch (e) {
                console.warn("ThemeService: Could not save to localStorage:", e);
            }
        },

        /**
         * Fire theme changed event to all registered callbacks
         * @private
         */
        _fireThemeChanged: function (sNewTheme) {
            var sOldTheme = sNewTheme === this.THEMES.DARK 
                ? this.THEMES.LIGHT 
                : this.THEMES.DARK;

            this._callbacks.forEach(function (fnCallback) {
                try {
                    fnCallback(sNewTheme, sOldTheme);
                } catch (e) {
                    console.error("ThemeService: Callback error:", e);
                }
            });
        }
    };

    // Initialize on load
    ThemeService.init();

    return ThemeService;
});

