/**
 * Client-side search using lunr.js
 */

var searchIndex = null;
var searchDocuments = null;

function initSearch(lang, baseUrl) {
    var input = document.getElementById('search-input');
    var resultsDiv = document.getElementById('search-results');
    if (!input || !resultsDiv) return;

    // Load search index
    fetch(baseUrl + '/search-index/' + lang + '.json')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            searchDocuments = {};
            data.documents.forEach(function(doc) {
                searchDocuments[doc.id] = doc;
            });

            searchIndex = lunr(function() {
                this.ref('id');
                this.field('title', { boost: 10 });
                this.field('topic', { boost: 5 });
                this.field('body');
                var self = this;
                data.documents.forEach(function(doc) { self.add(doc); });
            });

            // Check URL for initial query
            var params = new URLSearchParams(window.location.search);
            var q = params.get('q');
            if (q) {
                input.value = q;
                performSearch(q, resultsDiv);
            }
        })
        .catch(function(err) {
            console.error('Failed to load search index:', err);
        });

    // Debounced search on input
    var timeout;
    input.addEventListener('input', function() {
        clearTimeout(timeout);
        var value = this.value.trim();
        timeout = setTimeout(function() {
            if (value.length >= 2) {
                performSearch(value, resultsDiv);
                var url = new URL(window.location);
                url.searchParams.set('q', value);
                history.replaceState(null, '', url);
            } else {
                resultsDiv.innerHTML = '<div class="search-hint"><p>Enter a search term to find lessons across all topics.</p></div>';
            }
        }, 300);
    });

    // Submit sidebar search form to search page
    var sidebarForm = document.getElementById('search-form');
    if (sidebarForm) {
        sidebarForm.addEventListener('submit', function(e) {
            var sidebarInput = document.getElementById('search-sidebar-input');
            if (sidebarInput && sidebarInput.value.trim().length >= 2) {
                // Let the form submit normally to search.html
                return true;
            }
            e.preventDefault();
            return false;
        });
    }
}

function performSearch(query, resultsDiv) {
    if (!searchIndex) return;

    var results;
    try {
        results = searchIndex.search(query);
    } catch (e) {
        // If lunr query parsing fails, try as a simple term
        try {
            results = searchIndex.search(query.replace(/[:\~\^\+\-]/g, ' '));
        } catch (e2) {
            results = [];
        }
    }

    if (results.length === 0) {
        resultsDiv.innerHTML =
            '<p class="results-count">0 results for "' + escapeHtml(query) + '"</p>' +
            '<div class="no-results"><p>No results found. Try different keywords.</p></div>';
        return;
    }

    var html = '<p class="results-count">' + results.length + ' results for "' + escapeHtml(query) + '"</p>';
    html += '<div class="results-list">';
    results.forEach(function(result) {
        var doc = searchDocuments[result.ref];
        if (!doc) return;
        var snippet = generateSnippet(doc.body, query);
        html += '<a href="' + doc.url + '" class="result-item">' +
            '<div class="result-header">' +
            '<span class="result-topic">' + escapeHtml(doc.topic_display) + '</span>' +
            '<h3 class="result-title">' + escapeHtml(doc.title) + '</h3>' +
            '</div>' +
            '<p class="result-snippet">' + snippet + '</p>' +
            '</a>';
    });
    html += '</div>';
    resultsDiv.innerHTML = html;
}

function generateSnippet(body, query) {
    if (!body) return '';
    var terms = query.toLowerCase().split(/\s+/);
    var lowerBody = body.toLowerCase();
    var bestPos = 0;
    for (var i = 0; i < terms.length; i++) {
        var idx = lowerBody.indexOf(terms[i]);
        if (idx !== -1) { bestPos = idx; break; }
    }
    var start = Math.max(0, bestPos - 60);
    var end = Math.min(body.length, bestPos + 200);
    var snippet = (start > 0 ? '...' : '') + body.slice(start, end) + (end < body.length ? '...' : '');
    snippet = escapeHtml(snippet);
    for (var j = 0; j < terms.length; j++) {
        if (terms[j].length < 2) continue;
        var regex = new RegExp('(' + escapeRegex(terms[j]) + ')', 'gi');
        snippet = snippet.replace(regex, '<mark>$1</mark>');
    }
    return snippet;
}

function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
