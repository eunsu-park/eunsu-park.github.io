-- ============================================================================
-- PostgreSQL Full-Text Search Examples
-- ============================================================================
-- This script demonstrates full-text search capabilities in PostgreSQL,
-- including tsvector, tsquery, ranking, highlighting, and fuzzy matching.
--
-- Prerequisites:
--   - PostgreSQL 12+ (pg_trgm extension)
--   - CREATE EXTENSION privilege (for pg_trgm)
--
-- Usage:
--   psql -U postgres -d your_database -f 06_full_text_search.sql
-- ============================================================================

-- Clean up from previous runs
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS articles CASCADE;

-- ============================================================================
-- 1. Basic tsvector and tsquery
-- ============================================================================

-- Create a sample documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    -- Store pre-computed search vector for performance
    search_vector tsvector
);

-- Insert sample data
INSERT INTO documents (title, body) VALUES
    ('PostgreSQL Tutorial', 'PostgreSQL is a powerful open-source relational database system with advanced features.'),
    ('Full-Text Search Guide', 'Full-text search allows you to search natural language documents efficiently using inverted indexes.'),
    ('Database Performance', 'Optimizing database performance requires understanding indexes, query planning, and proper schema design.'),
    ('Advanced SQL Techniques', 'Learn advanced SQL techniques including window functions, CTEs, and recursive queries.'),
    ('Python and PostgreSQL', 'Connecting Python applications to PostgreSQL databases using psycopg2 and SQLAlchemy.');

-- Convert text to tsvector (tokenize and normalize)
SELECT to_tsvector('english', 'PostgreSQL is a powerful database system');
-- Output: 'databas':5 'postgresql':1 'power':4 'system':6

-- Create tsquery (search pattern)
SELECT to_tsquery('english', 'database & system');
-- Output: 'databas' & 'system'

-- Match tsvector against tsquery
SELECT to_tsvector('english', 'PostgreSQL is a powerful database system')
       @@ to_tsquery('english', 'database & system') AS matches;
-- Output: true

-- ============================================================================
-- 2. Creating a GIN Index for Fast Search
-- ============================================================================

-- Update search_vector column with combined title and body
UPDATE documents
SET search_vector =
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(body, '')), 'B');

-- Create GIN (Generalized Inverted Index) for fast full-text search
CREATE INDEX idx_documents_search ON documents USING GIN(search_vector);

-- Basic search query
SELECT id, title
FROM documents
WHERE search_vector @@ to_tsquery('english', 'database');

-- Search with AND operator
SELECT id, title
FROM documents
WHERE search_vector @@ to_tsquery('english', 'database & performance');

-- Search with OR operator
SELECT id, title
FROM documents
WHERE search_vector @@ to_tsquery('english', 'PostgreSQL | Python');

-- Search with negation
SELECT id, title
FROM documents
WHERE search_vector @@ to_tsquery('english', 'database & !performance');

-- Phrase search (use <-> for word distance)
SELECT id, title
FROM documents
WHERE search_vector @@ to_tsquery('english', 'full <-> text <-> search');

-- ============================================================================
-- 3. Weighted Search (Priority Ranking)
-- ============================================================================

-- Assign different weights to different parts of the document
-- Weight 'A' (highest) for title, 'B' for body
-- Weights: A=1.0, B=0.4, C=0.2, D=0.1

SELECT
    id,
    title,
    -- Show which parts matched
    ts_debug('english', title) AS title_tokens
FROM documents
WHERE search_vector @@ to_tsquery('english', 'PostgreSQL')
LIMIT 1;

-- ============================================================================
-- 4. Relevance Ranking with ts_rank
-- ============================================================================

-- Rank search results by relevance
SELECT
    id,
    title,
    ts_rank(search_vector, query) AS rank
FROM documents, to_tsquery('english', 'database') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- ts_rank_cd (cover density ranking) - considers proximity of terms
SELECT
    id,
    title,
    ts_rank_cd(search_vector, query, 32) AS rank_cd
FROM documents, to_tsquery('english', 'database & system') query
WHERE search_vector @@ query
ORDER BY rank_cd DESC;

-- Custom ranking with weights
SELECT
    id,
    title,
    ts_rank(
        search_vector,
        query,
        -- Normalization flags: 1=length, 2=log(length), 4=harmonic distance
        1  -- Normalize by document length
    ) AS rank
FROM documents, to_tsquery('english', 'PostgreSQL | database') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- ============================================================================
-- 5. Result Highlighting with ts_headline
-- ============================================================================

-- Highlight matching terms in search results
SELECT
    id,
    title,
    ts_headline(
        'english',
        body,
        to_tsquery('english', 'database & system'),
        'StartSel=<b>, StopSel=</b>, MaxWords=50, MinWords=20'
    ) AS snippet
FROM documents
WHERE search_vector @@ to_tsquery('english', 'database & system');

-- Highlight with custom options
SELECT
    id,
    ts_headline(
        'english',
        title || ' ' || body,
        to_tsquery('english', 'PostgreSQL'),
        'StartSel=**, StopSel=**, MaxFragments=2, FragmentDelimiter=...'
    ) AS highlighted
FROM documents
WHERE search_vector @@ to_tsquery('english', 'PostgreSQL');

-- ============================================================================
-- 6. Advanced: plainto_tsquery and websearch_to_tsquery
-- ============================================================================

-- plainto_tsquery: Simple query parser (automatic AND between words)
SELECT
    id,
    title,
    ts_rank(search_vector, query) AS rank
FROM documents, plainto_tsquery('english', 'database performance') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- websearch_to_tsquery: Google-like search syntax (PostgreSQL 11+)
-- Supports: "phrase", OR, -, quotes
SELECT
    id,
    title,
    ts_rank(search_vector, query) AS rank
FROM documents, websearch_to_tsquery('english', '"full-text search" OR PostgreSQL -Python') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- ============================================================================
-- 7. Fuzzy Matching with pg_trgm
-- ============================================================================

-- Enable pg_trgm extension for trigram-based fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create articles table for fuzzy search demo
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL
);

INSERT INTO articles (title, content) VALUES
    ('Introduction to Databases', 'Learn about relational and NoSQL databases.'),
    ('Database Design Patterns', 'Common patterns for designing scalable database schemas.'),
    ('PostgreSQL Administration', 'Essential PostgreSQL administration tasks and tools.'),
    ('Data Warehousing Basics', 'Introduction to data warehousing concepts and star schemas.');

-- Create GIN index for trigram matching
CREATE INDEX idx_articles_title_trgm ON articles USING GIN(title gin_trgm_ops);

-- Similarity search (0.0 = no match, 1.0 = perfect match)
SELECT
    title,
    similarity(title, 'Databse Design') AS sim
FROM articles
WHERE title % 'Databse Design'  -- % operator checks if similarity > threshold
ORDER BY sim DESC;

-- Set similarity threshold (default is 0.3)
SET pg_trgm.similarity_threshold = 0.2;

SELECT
    title,
    similarity(title, 'Postgre') AS sim
FROM articles
WHERE title % 'Postgre'
ORDER BY sim DESC;

-- Word similarity (for substring matching)
SELECT
    title,
    word_similarity('Admin', title) AS word_sim
FROM articles
WHERE 'Admin' <% title  -- <% operator for word similarity
ORDER BY word_sim DESC;

-- Combine full-text search with fuzzy matching
SELECT
    a.title,
    ts_rank(to_tsvector('english', a.content), query) AS fts_rank,
    similarity(a.title, 'Databse') AS fuzzy_sim
FROM articles a, to_tsquery('english', 'database') query
WHERE
    to_tsvector('english', a.content) @@ query
    OR a.title % 'Databse'
ORDER BY fts_rank DESC, fuzzy_sim DESC;

-- ============================================================================
-- 8. Automatic Search Vector Update with Trigger
-- ============================================================================

-- Create trigger function to auto-update search_vector
CREATE OR REPLACE FUNCTION documents_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.body, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to documents table
DROP TRIGGER IF EXISTS tsvector_update ON documents;
CREATE TRIGGER tsvector_update
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_search_vector_update();

-- Test trigger
INSERT INTO documents (title, body) VALUES
    ('Trigger Test', 'This document tests automatic search vector updates.');

SELECT title, search_vector
FROM documents
WHERE title = 'Trigger Test';

-- ============================================================================
-- 9. Multi-language Full-Text Search
-- ============================================================================

-- PostgreSQL supports many languages (run \dF to see available configurations)
-- Examples: 'simple', 'english', 'french', 'german', 'spanish', etc.

-- Spanish text search
SELECT to_tsvector('spanish', 'Los documentos están escritos en español');

-- Mixed language table
CREATE TABLE multilang_docs (
    id SERIAL PRIMARY KEY,
    lang TEXT NOT NULL,
    content TEXT NOT NULL,
    search_vector tsvector
);

INSERT INTO multilang_docs (lang, content) VALUES
    ('english', 'The quick brown fox jumps over the lazy dog'),
    ('french', 'Le renard brun rapide saute par-dessus le chien paresseux'),
    ('german', 'Der schnelle braune Fuchs springt über den faulen Hund');

-- Update with language-specific search vectors
UPDATE multilang_docs
SET search_vector = to_tsvector(lang::regconfig, content);

-- Search in specific language
SELECT lang, content
FROM multilang_docs
WHERE search_vector @@ to_tsquery('french', 'renard');

-- ============================================================================
-- 10. Performance Tips
-- ============================================================================

-- Analyze search performance
EXPLAIN ANALYZE
SELECT id, title, ts_rank(search_vector, query) AS rank
FROM documents, to_tsquery('english', 'database & performance') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Vacuum and analyze to update statistics
VACUUM ANALYZE documents;

-- Monitor index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'documents';

-- ============================================================================
-- Summary
-- ============================================================================
-- Key takeaways:
-- 1. tsvector: Tokenized, normalized text representation
-- 2. tsquery: Search pattern with operators (&, |, !, <->)
-- 3. GIN index: Fast full-text search (essential for production)
-- 4. Weights: Prioritize different document parts (A > B > C > D)
-- 5. ts_rank: Relevance scoring for result ordering
-- 6. ts_headline: Highlight matching terms in results
-- 7. pg_trgm: Fuzzy matching for typo tolerance
-- 8. Triggers: Auto-update search vectors on INSERT/UPDATE
-- 9. Multi-language: Support for 20+ languages
-- 10. Performance: Use GIN indexes, VACUUM ANALYZE, monitor stats
-- ============================================================================
