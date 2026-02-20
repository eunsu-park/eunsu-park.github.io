/**
 * Simple Hash-Based SPA Router
 *
 * Features:
 * - Route registration with path patterns
 * - Dynamic route parameters (e.g., /user/:id)
 * - Hash-based navigation (#/path)
 * - 404 handling
 * - Navigation guards (before/after hooks)
 * - History API support
 */

class Router {
    constructor() {
        this.routes = {};
        this.currentRoute = null;
        this.notFoundHandler = null;
        this.beforeHooks = [];
        this.afterHooks = [];

        // Listen for hash changes
        window.addEventListener('hashchange', () => this.handleRoute());
        window.addEventListener('load', () => this.handleRoute());
    }

    /**
     * Register a route with its handler
     * @param {string} path - Route path (e.g., '/', '/user/:id')
     * @param {Function} handler - Function that returns HTML content
     */
    addRoute(path, handler) {
        this.routes[path] = {
            pattern: this.pathToRegex(path),
            handler: handler,
            path: path
        };
    }

    /**
     * Convert path pattern to regex for matching
     * Supports dynamic parameters like :id, :name
     */
    pathToRegex(path) {
        // Escape special regex characters except :
        const pattern = path
            .replace(/\//g, '\\/')
            .replace(/:\w+/g, '([^/]+)');
        return new RegExp(`^${pattern}$`);
    }

    /**
     * Extract parameters from URL path
     * @param {string} route - Route pattern with parameters
     * @param {string} path - Actual path from URL
     * @returns {Object} Parameters object
     */
    extractParams(route, path) {
        const params = {};
        const routeParts = route.split('/');
        const pathParts = path.split('/');

        routeParts.forEach((part, i) => {
            if (part.startsWith(':')) {
                const paramName = part.slice(1);
                params[paramName] = decodeURIComponent(pathParts[i]);
            }
        });

        return params;
    }

    /**
     * Navigate to a new route programmatically
     */
    navigate(path) {
        window.location.hash = path;
    }

    /**
     * Handle route changes
     */
    async handleRoute() {
        const path = window.location.hash.slice(1) || '/';

        // Execute before hooks
        for (const hook of this.beforeHooks) {
            const result = await hook(path);
            if (result === false) return; // Cancel navigation
        }

        // Find matching route
        let matchedRoute = null;
        let params = {};

        for (const [routePath, route] of Object.entries(this.routes)) {
            if (route.pattern.test(path)) {
                matchedRoute = route;
                params = this.extractParams(routePath, path);
                break;
            }
        }

        // Render content
        const appElement = document.getElementById('app');

        if (matchedRoute) {
            this.currentRoute = { path, params };
            appElement.innerHTML = await matchedRoute.handler(params);
            this.updateActiveLinks(path);
        } else if (this.notFoundHandler) {
            appElement.innerHTML = await this.notFoundHandler();
        } else {
            appElement.innerHTML = '<h1>404 - Page Not Found</h1>';
        }

        // Add page transition animation
        appElement.classList.remove('page-enter');
        void appElement.offsetWidth; // Trigger reflow
        appElement.classList.add('page-enter');

        // Execute after hooks
        for (const hook of this.afterHooks) {
            await hook(path, params);
        }
    }

    /**
     * Update active state of navigation links
     */
    updateActiveLinks(currentPath) {
        document.querySelectorAll('.nav-link').forEach(link => {
            const href = link.getAttribute('href').slice(1); // Remove #
            link.classList.toggle('active', href === currentPath);
        });
    }

    /**
     * Set 404 handler
     */
    setNotFound(handler) {
        this.notFoundHandler = handler;
    }

    /**
     * Add navigation guard (before route change)
     */
    beforeEach(hook) {
        this.beforeHooks.push(hook);
    }

    /**
     * Add hook after route change
     */
    afterEach(hook) {
        this.afterHooks.push(hook);
    }
}

// Export router instance
export default new Router();
