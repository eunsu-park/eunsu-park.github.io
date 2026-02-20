/**
 * Main Application Logic
 *
 * Initializes router, defines page components, and manages state
 */

import router from './router.js';

// Simple State Management
const state = {
    user: null,
    visitCount: 0,

    setUser(user) {
        this.user = user;
    },

    incrementVisit() {
        this.visitCount++;
    }
};

// Page Components

/**
 * Home Page Component
 */
function HomePage() {
    return `
        <div class="page-enter">
            <h1>Welcome to SPA Router Demo</h1>
            <p>This is a demonstration of a simple single-page application using hash-based routing.</p>

            <div class="card">
                <h2>Features</h2>
                <ul>
                    <li>Hash-based routing (#/path)</li>
                    <li>Dynamic route parameters</li>
                    <li>Page transition animations</li>
                    <li>Mobile responsive design</li>
                    <li>Navigation guards</li>
                    <li>404 error handling</li>
                </ul>
            </div>

            <div class="card">
                <h2>Getting Started</h2>
                <p>Use the navigation bar above to explore different pages:</p>
                <ul>
                    <li><strong>Home:</strong> You are here!</li>
                    <li><strong>About:</strong> Learn more about this project</li>
                    <li><strong>Contact:</strong> Fill out a contact form</li>
                    <li><strong>User Profile:</strong> View dynamic route parameters</li>
                </ul>
            </div>

            <div class="card">
                <h3>Visit Counter</h3>
                <p>This page has been loaded <strong>${state.visitCount}</strong> time(s).</p>
            </div>

            <a href="#/about" class="btn">Learn More</a>
        </div>
    `;
}

/**
 * About Page Component
 */
function AboutPage() {
    return `
        <div class="page-enter">
            <h1>About This Project</h1>

            <div class="card">
                <h2>What is a Single Page Application?</h2>
                <p>
                    A Single Page Application (SPA) is a web application that loads a single HTML page
                    and dynamically updates content as the user interacts with the app. This provides
                    a more fluid user experience similar to desktop applications.
                </p>
            </div>

            <div class="card">
                <h2>Hash-Based Routing</h2>
                <p>
                    This implementation uses hash-based routing (<code>#/path</code>), which:
                </p>
                <ul>
                    <li>Works without server configuration</li>
                    <li>Doesn't trigger page reloads</li>
                    <li>Maintains browser history</li>
                    <li>Enables back/forward navigation</li>
                </ul>
            </div>

            <div class="card">
                <h2>Technology Stack</h2>
                <ul>
                    <li><strong>HTML5:</strong> Semantic markup</li>
                    <li><strong>CSS3:</strong> Custom properties, animations, flexbox</li>
                    <li><strong>JavaScript ES6+:</strong> Modules, classes, async/await</li>
                    <li><strong>No Framework:</strong> Pure vanilla JavaScript</li>
                </ul>
            </div>

            <a href="#/contact" class="btn btn-secondary">Get in Touch</a>
        </div>
    `;
}

/**
 * Contact Page Component
 */
function ContactPage() {
    // Add form submission handler after render
    setTimeout(() => {
        const form = document.getElementById('contact-form');
        if (form) {
            form.addEventListener('submit', handleContactSubmit);
        }
    }, 0);

    return `
        <div class="page-enter">
            <h1>Contact Us</h1>
            <p>Have questions? Fill out the form below and we'll get back to you.</p>

            <form id="contact-form" class="card">
                <div class="form-group">
                    <label class="form-label" for="name">Name:</label>
                    <input type="text" id="name" name="name" class="form-control" required>
                </div>

                <div class="form-group">
                    <label class="form-label" for="email">Email:</label>
                    <input type="email" id="email" name="email" class="form-control" required>
                </div>

                <div class="form-group">
                    <label class="form-label" for="message">Message:</label>
                    <textarea id="message" name="message" rows="5" class="form-control" required></textarea>
                </div>

                <button type="submit" class="btn">Send Message</button>
            </form>

            <div id="form-result" style="margin-top: 1rem;"></div>
        </div>
    `;
}

/**
 * User Profile Page Component (Dynamic Route)
 */
function UserProfilePage(params) {
    const userId = params.id || 'unknown';

    // Simulate user data
    const userData = {
        '123': { name: 'John Doe', email: 'john@example.com', role: 'Developer' },
        '456': { name: 'Jane Smith', email: 'jane@example.com', role: 'Designer' },
        'unknown': { name: 'Guest User', email: 'N/A', role: 'Visitor' }
    };

    const user = userData[userId] || userData['unknown'];

    return `
        <div class="page-enter">
            <h1>User Profile</h1>

            <div class="card">
                <h2>${user.name}</h2>
                <p><strong>User ID:</strong> ${userId}</p>
                <p><strong>Email:</strong> ${user.email}</p>
                <p><strong>Role:</strong> ${user.role}</p>
            </div>

            <div class="card">
                <h3>Dynamic Route Parameters</h3>
                <p>This page demonstrates dynamic routing. The URL pattern is <code>/user/:id</code></p>
                <p>Current route: <code>/user/${userId}</code></p>
                <p>Try different user IDs:</p>
                <ul>
                    <li><a href="#/user/123">/user/123</a> - John Doe</li>
                    <li><a href="#/user/456">/user/456</a> - Jane Smith</li>
                    <li><a href="#/user/789">/user/789</a> - Unknown user</li>
                </ul>
            </div>

            <a href="#/" class="btn">Back to Home</a>
        </div>
    `;
}

/**
 * 404 Not Found Page
 */
function NotFoundPage() {
    return `
        <div class="page-enter">
            <h1>404 - Page Not Found</h1>
            <p>The page you're looking for doesn't exist.</p>
            <a href="#/" class="btn">Go Home</a>
        </div>
    `;
}

// Event Handlers

function handleContactSubmit(e) {
    e.preventDefault();

    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        message: document.getElementById('message').value
    };

    // Simulate form submission
    const resultDiv = document.getElementById('form-result');
    resultDiv.innerHTML = `
        <div class="card" style="border-left-color: #2ecc71;">
            <h3>Thank you, ${formData.name}!</h3>
            <p>Your message has been received. We'll respond to ${formData.email} soon.</p>
        </div>
    `;

    // Reset form
    document.getElementById('contact-form').reset();
}

// Router Configuration

// Register routes
router.addRoute('/', HomePage);
router.addRoute('/about', AboutPage);
router.addRoute('/contact', ContactPage);
router.addRoute('/user/:id', UserProfilePage);

// Set 404 handler
router.setNotFound(NotFoundPage);

// Navigation guards
router.beforeEach((path) => {
    console.log('Navigating to:', path);
    state.incrementVisit();
    return true; // Continue navigation
});

router.afterEach((path, params) => {
    console.log('Navigation complete:', path, params);
    // Update page title
    const titles = {
        '/': 'Home',
        '/about': 'About',
        '/contact': 'Contact',
    };
    document.title = `SPA Example - ${titles[path] || 'Page'}`;
});

console.log('App initialized successfully');
