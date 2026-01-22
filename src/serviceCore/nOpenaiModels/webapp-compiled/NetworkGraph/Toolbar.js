/**
 * NetworkGraph Header Toolbar
 * Matches SAP Fiori specification for Network Graph toolbars
 * Includes: Title, Search, Legend Toggle, Zoom Controls, Fullscreen
 */
import { SAP_COLORS } from './types';
// ============================================================================
// Toolbar Class
// ============================================================================
export class NetworkGraphToolbar {
    constructor(container, config = {}) {
        // State
        this.isLegendVisible = false;
        this.isFullscreen = false;
        this.currentZoom = 100;
        // Event callbacks
        this.onSearch = null;
        this.onZoomIn = null;
        this.onZoomOut = null;
        this.onFitToView = null;
        this.onFullscreen = null;
        this.onLegendToggle = null;
        this.container = container;
        this.config = { ...NetworkGraphToolbar.DEFAULT_CONFIG, ...config };
        this.createToolbar();
        this.setupEventListeners();
    }
    // ========================================================================
    // DOM Creation
    // ========================================================================
    createToolbar() {
        this.toolbarElement = document.createElement('div');
        this.toolbarElement.className = 'network-graph-toolbar';
        this.toolbarElement.style.cssText = `
            display: flex;
            align-items: center;
            height: 48px;
            padding: 0 16px;
            background: ${SAP_COLORS.backgroundAlt};
            border-bottom: 1px solid ${SAP_COLORS.border};
            font-family: "72", Arial, Helvetica, sans-serif;
            box-sizing: border-box;
        `;
        // (A) Title
        this.titleElement = this.createTitle();
        this.toolbarElement.appendChild(this.titleElement);
        // Spacer
        const spacer = document.createElement('div');
        spacer.style.flex = '1';
        this.toolbarElement.appendChild(spacer);
        // (B) Search Field
        if (this.config.showSearch) {
            const searchContainer = this.createSearchField();
            this.toolbarElement.appendChild(searchContainer);
        }
        // Separator
        this.toolbarElement.appendChild(this.createSeparator());
        // (C) Legend Toggle
        if (this.config.showLegend) {
            this.legendButton = this.createLegendButton();
            this.toolbarElement.appendChild(this.legendButton);
        }
        // Separator
        this.toolbarElement.appendChild(this.createSeparator());
        // Zoom Controls (D, E, F, G)
        if (this.config.showZoomControls) {
            const zoomControls = this.createZoomControls();
            this.toolbarElement.appendChild(zoomControls);
        }
        // Separator
        this.toolbarElement.appendChild(this.createSeparator());
        // (I) Fullscreen Toggle
        if (this.config.showFullscreen) {
            this.fullscreenButton = this.createFullscreenButton();
            this.toolbarElement.appendChild(this.fullscreenButton);
        }
        this.container.insertBefore(this.toolbarElement, this.container.firstChild);
    }
    createTitle() {
        const title = document.createElement('h2');
        title.className = 'toolbar-title';
        title.textContent = this.config.title || 'Network Graph';
        title.style.cssText = `
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            color: ${SAP_COLORS.text};
            white-space: nowrap;
        `;
        return title;
    }
    createSearchField() {
        const container = document.createElement('div');
        container.className = 'toolbar-search';
        container.style.cssText = `
            position: relative;
            margin-right: 12px;
        `;
        // Search input
        this.searchInput = document.createElement('input');
        this.searchInput.type = 'text';
        this.searchInput.placeholder = 'Search nodes...';
        this.searchInput.className = 'toolbar-search-input';
        this.searchInput.style.cssText = `
            width: 200px;
            height: 32px;
            padding: 0 32px 0 12px;
            border: 1px solid ${SAP_COLORS.border};
            border-radius: 4px;
            font-family: inherit;
            font-size: 14px;
            color: ${SAP_COLORS.text};
            background: ${SAP_COLORS.backgroundAlt};
            outline: none;
            transition: border-color 0.15s ease;
        `;
        container.appendChild(this.searchInput);
        // Search icon (magnifying glass)
        const searchIcon = document.createElement('span');
        searchIcon.innerHTML = '🔍';
        searchIcon.style.cssText = `
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 14px;
            pointer-events: none;
            opacity: 0.6;
        `;
        container.appendChild(searchIcon);
        // Suggestions dropdown
        this.searchSuggestions = document.createElement('div');
        this.searchSuggestions.className = 'toolbar-search-suggestions';
        this.searchSuggestions.style.cssText = `
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            max-height: 200px;
            overflow-y: auto;
            background: ${SAP_COLORS.backgroundAlt};
            border: 1px solid ${SAP_COLORS.border};
            border-top: none;
            border-radius: 0 0 4px 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            display: none;
            z-index: 1000;
        `;
        container.appendChild(this.searchSuggestions);
        return container;
    }
    createLegendButton() {
        const button = this.createToolbarButton('☰', 'Toggle Legend');
        button.setAttribute('aria-pressed', 'false');
        return button;
    }
    createZoomControls() {
        const container = document.createElement('div');
        container.className = 'toolbar-zoom-controls';
        container.style.cssText = `
            display: flex;
            align-items: center;
            gap: 4px;
        `;
        // (D) Zoom In Button
        this.zoomInButton = this.createToolbarButton('+', 'Zoom In');
        container.appendChild(this.zoomInButton);
        // (E) Zoom Level Display
        this.zoomLevelDisplay = document.createElement('span');
        this.zoomLevelDisplay.className = 'toolbar-zoom-level';
        this.zoomLevelDisplay.textContent = '100%';
        this.zoomLevelDisplay.style.cssText = `
            min-width: 48px;
            text-align: center;
            font-size: 13px;
            color: ${SAP_COLORS.text};
            padding: 0 4px;
        `;
        container.appendChild(this.zoomLevelDisplay);
        // (F) Zoom Out Button
        this.zoomOutButton = this.createToolbarButton('−', 'Zoom Out');
        container.appendChild(this.zoomOutButton);
        // (G) Fit to Viewport Button
        this.fitButton = this.createToolbarButton('⛶', 'Fit to Viewport');
        container.appendChild(this.fitButton);
        return container;
    }
    createFullscreenButton() {
        const button = this.createToolbarButton('⛶', 'Toggle Fullscreen');
        button.className += ' toolbar-fullscreen-btn';
        return button;
    }
    createToolbarButton(icon, tooltip) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'toolbar-button';
        button.textContent = icon;
        button.title = tooltip;
        button.setAttribute('aria-label', tooltip);
        button.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border: 1px solid transparent;
            border-radius: 4px;
            background: transparent;
            color: ${SAP_COLORS.text};
            font-size: 16px;
            cursor: pointer;
            transition: all 0.15s ease;
        `;
        // Hover effect
        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = SAP_COLORS.background;
            button.style.borderColor = SAP_COLORS.border;
            button.style.color = SAP_COLORS.brand;
        });
        button.addEventListener('mouseleave', () => {
            if (!button.classList.contains('active')) {
                button.style.backgroundColor = 'transparent';
                button.style.borderColor = 'transparent';
                button.style.color = SAP_COLORS.text;
            }
        });
        return button;
    }
    createSeparator() {
        const sep = document.createElement('div');
        sep.className = 'toolbar-separator';
        sep.style.cssText = `
            width: 1px;
            height: 24px;
            background: ${SAP_COLORS.border};
            margin: 0 12px;
        `;
        return sep;
    }
    // ========================================================================
    // Event Listeners
    // ========================================================================
    setupEventListeners() {
        // Search input
        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => {
                const query = e.target.value;
                if (this.onSearch) {
                    this.onSearch(query);
                }
            });
            this.searchInput.addEventListener('focus', () => {
                this.searchInput.style.borderColor = SAP_COLORS.brand;
                if (this.searchSuggestions.children.length > 0) {
                    this.searchSuggestions.style.display = 'block';
                }
            });
            this.searchInput.addEventListener('blur', () => {
                this.searchInput.style.borderColor = SAP_COLORS.border;
                // Delay hiding to allow clicking on suggestions
                setTimeout(() => {
                    this.searchSuggestions.style.display = 'none';
                }, 200);
            });
        }
        // Legend toggle
        if (this.legendButton) {
            this.legendButton.addEventListener('click', () => {
                this.isLegendVisible = !this.isLegendVisible;
                this.updateButtonActiveState(this.legendButton, this.isLegendVisible);
                if (this.onLegendToggle) {
                    this.onLegendToggle(this.isLegendVisible);
                }
            });
        }
        // Zoom controls
        if (this.zoomInButton) {
            this.zoomInButton.addEventListener('click', () => {
                if (this.onZoomIn)
                    this.onZoomIn();
            });
        }
        if (this.zoomOutButton) {
            this.zoomOutButton.addEventListener('click', () => {
                if (this.onZoomOut)
                    this.onZoomOut();
            });
        }
        if (this.fitButton) {
            this.fitButton.addEventListener('click', () => {
                if (this.onFitToView)
                    this.onFitToView();
            });
        }
        // Fullscreen toggle
        if (this.fullscreenButton) {
            this.fullscreenButton.addEventListener('click', () => {
                this.isFullscreen = !this.isFullscreen;
                this.updateButtonActiveState(this.fullscreenButton, this.isFullscreen);
                if (this.onFullscreen) {
                    this.onFullscreen(this.isFullscreen);
                }
            });
        }
    }
    updateButtonActiveState(button, active) {
        if (active) {
            button.classList.add('active');
            button.style.backgroundColor = SAP_COLORS.brand;
            button.style.color = SAP_COLORS.backgroundAlt;
            button.style.borderColor = SAP_COLORS.brand;
            button.setAttribute('aria-pressed', 'true');
        }
        else {
            button.classList.remove('active');
            button.style.backgroundColor = 'transparent';
            button.style.color = SAP_COLORS.text;
            button.style.borderColor = 'transparent';
            button.setAttribute('aria-pressed', 'false');
        }
    }
    // ========================================================================
    // Public Methods
    // ========================================================================
    setTitle(title) {
        if (this.titleElement) {
            this.titleElement.textContent = title;
        }
    }
    setZoomLevel(percent) {
        this.currentZoom = Math.round(percent);
        if (this.zoomLevelDisplay) {
            this.zoomLevelDisplay.textContent = `${this.currentZoom}%`;
        }
    }
    setSearchSuggestions(suggestions) {
        if (!this.searchSuggestions)
            return;
        this.searchSuggestions.innerHTML = '';
        if (suggestions.length === 0) {
            this.searchSuggestions.style.display = 'none';
            return;
        }
        for (const suggestion of suggestions) {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.textContent = suggestion;
            item.style.cssText = `
                padding: 8px 12px;
                cursor: pointer;
                font-size: 14px;
                color: ${SAP_COLORS.text};
                transition: background 0.15s ease;
            `;
            item.addEventListener('mouseenter', () => {
                item.style.backgroundColor = SAP_COLORS.background;
            });
            item.addEventListener('mouseleave', () => {
                item.style.backgroundColor = 'transparent';
            });
            item.addEventListener('click', () => {
                this.searchInput.value = suggestion;
                this.searchSuggestions.style.display = 'none';
                if (this.onSearch) {
                    this.onSearch(suggestion);
                }
            });
            this.searchSuggestions.appendChild(item);
        }
        // Show if search input is focused
        if (document.activeElement === this.searchInput) {
            this.searchSuggestions.style.display = 'block';
        }
    }
    on(event, callback) {
        switch (event) {
            case 'search':
                this.onSearch = callback;
                break;
            case 'zoomIn':
                this.onZoomIn = callback;
                break;
            case 'zoomOut':
                this.onZoomOut = callback;
                break;
            case 'fitToView':
                this.onFitToView = callback;
                break;
            case 'fullscreen':
                this.onFullscreen = callback;
                break;
            case 'legendToggle':
                this.onLegendToggle = callback;
                break;
        }
    }
    getSearchQuery() {
        return this.searchInput?.value || '';
    }
    isLegendShown() {
        return this.isLegendVisible;
    }
    isFullscreenEnabled() {
        return this.isFullscreen;
    }
    getZoomLevel() {
        return this.currentZoom;
    }
    // ========================================================================
    // Cleanup
    // ========================================================================
    destroy() {
        if (this.toolbarElement && this.toolbarElement.parentNode) {
            this.toolbarElement.parentNode.removeChild(this.toolbarElement);
        }
        // Clear callbacks
        this.onSearch = null;
        this.onZoomIn = null;
        this.onZoomOut = null;
        this.onFitToView = null;
        this.onFullscreen = null;
        this.onLegendToggle = null;
    }
}
// Default config
NetworkGraphToolbar.DEFAULT_CONFIG = {
    title: 'Network Graph',
    showSearch: true,
    showLegend: true,
    showZoomControls: true,
    showFullscreen: true
};
//# sourceMappingURL=Toolbar.js.map