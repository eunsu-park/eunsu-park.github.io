/**
 * Study Materials - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    initSidebar();
});

/**
 * Theme Management
 */
function initTheme() {
    var themeToggle = document.getElementById('theme-toggle');
    var html = document.documentElement;

    var savedTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            var currentTheme = html.getAttribute('data-theme');
            var newTheme = currentTheme === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
}

/**
 * Sidebar Management (Mobile)
 */
function initSidebar() {
    var sidebar = document.getElementById('sidebar');
    var menuToggle = document.getElementById('menu-toggle');
    var sidebarToggle = document.getElementById('sidebar-toggle');

    if (menuToggle) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.add('open');
        });
    }

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.remove('open');
        });
    }

    document.addEventListener('click', function(e) {
        if (sidebar && sidebar.classList.contains('open')) {
            if (!sidebar.contains(e.target) && e.target !== menuToggle) {
                sidebar.classList.remove('open');
            }
        }
    });
}

/**
 * Copy to Clipboard utility
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        return navigator.clipboard.writeText(text);
    }
    var textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    return Promise.resolve();
}

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', function(e) {
    // Cmd/Ctrl + K for search
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        var searchInput = document.getElementById('search-sidebar-input');
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Escape to close sidebar on mobile
    if (e.key === 'Escape') {
        var sidebar = document.getElementById('sidebar');
        if (sidebar && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    }
});
