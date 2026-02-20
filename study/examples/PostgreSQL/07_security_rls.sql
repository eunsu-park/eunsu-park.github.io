-- ============================================================================
-- PostgreSQL Security and Row-Level Security (RLS) Examples
-- ============================================================================
-- This script demonstrates PostgreSQL security features including:
-- - Role management and privileges
-- - Row-Level Security (RLS) policies
-- - Multi-tenant data isolation
-- - Audit logging with triggers
-- - Encryption and hashing with pgcrypto
--
-- Prerequisites:
--   - PostgreSQL 9.5+ (for RLS)
--   - Superuser or role creation privileges
--   - pgcrypto extension
--
-- Usage:
--   psql -U postgres -d your_database -f 07_security_rls.sql
--
-- IMPORTANT: Run this script as a superuser (postgres) to create roles.
-- ============================================================================

-- Clean up from previous runs
DROP TABLE IF EXISTS projects CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS sensitive_data CASCADE;

-- Drop roles if they exist (ignore errors if they don't)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_admin') THEN
        REASSIGN OWNED BY app_admin TO postgres;
        DROP OWNED BY app_admin;
        DROP ROLE app_admin;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        REASSIGN OWNED BY app_user TO postgres;
        DROP OWNED BY app_user;
        DROP ROLE app_user;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_readonly') THEN
        REASSIGN OWNED BY app_readonly TO postgres;
        DROP OWNED BY app_readonly;
        DROP ROLE app_readonly;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_a_user') THEN
        REASSIGN OWNED BY tenant_a_user TO postgres;
        DROP OWNED BY tenant_a_user;
        DROP ROLE tenant_a_user;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tenant_b_user') THEN
        REASSIGN OWNED BY tenant_b_user TO postgres;
        DROP OWNED BY tenant_b_user;
        DROP ROLE tenant_b_user;
    END IF;
END
$$;

-- ============================================================================
-- 1. Role Management and Privilege Levels
-- ============================================================================

-- Create roles with different privilege levels
CREATE ROLE app_admin WITH LOGIN PASSWORD 'admin_password_123';
CREATE ROLE app_user WITH LOGIN PASSWORD 'user_password_123';
CREATE ROLE app_readonly WITH LOGIN PASSWORD 'readonly_password_123';

-- Grant connection privilege to database
GRANT CONNECT ON DATABASE postgres TO app_admin, app_user, app_readonly;

-- Create a schema for application tables
CREATE SCHEMA IF NOT EXISTS app;
GRANT USAGE ON SCHEMA app TO app_admin, app_user, app_readonly;

-- Create a sample table
CREATE TABLE app.projects (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    owner TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 2. GRANT and REVOKE Privileges
-- ============================================================================

-- Admin: Full access (SELECT, INSERT, UPDATE, DELETE)
GRANT ALL PRIVILEGES ON TABLE app.projects TO app_admin;
GRANT USAGE, SELECT ON SEQUENCE app.projects_id_seq TO app_admin;

-- User: Read and write access (SELECT, INSERT, UPDATE)
GRANT SELECT, INSERT, UPDATE ON TABLE app.projects TO app_user;
GRANT USAGE, SELECT ON SEQUENCE app.projects_id_seq TO app_user;

-- Readonly: Read-only access (SELECT)
GRANT SELECT ON TABLE app.projects TO app_readonly;

-- Grant schema-level privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA app
    GRANT ALL PRIVILEGES ON TABLES TO app_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA app
    GRANT SELECT, INSERT, UPDATE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA app
    GRANT SELECT ON TABLES TO app_readonly;

-- Revoke specific privilege (e.g., prevent DELETE for app_user)
REVOKE DELETE ON TABLE app.projects FROM app_user;

-- Check current privileges
SELECT
    grantee,
    privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'app' AND table_name = 'projects'
ORDER BY grantee, privilege_type;

-- ============================================================================
-- 3. Row-Level Security (RLS) Basics
-- ============================================================================

-- Create a documents table with RLS
CREATE TABLE app.documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    owner TEXT NOT NULL,  -- Stores the role name of the owner
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert sample data
INSERT INTO app.documents (title, content, owner) VALUES
    ('Admin Document', 'Confidential admin content', 'app_admin'),
    ('User Document 1', 'User content 1', 'app_user'),
    ('User Document 2', 'User content 2', 'app_user'),
    ('Readonly Document', 'Public content', 'app_readonly');

-- Enable Row-Level Security on the table
ALTER TABLE app.documents ENABLE ROW LEVEL SECURITY;

-- Without policies, no non-owner can access rows (default deny)
-- Superusers and table owners bypass RLS by default

-- Create policy: Users can only see their own documents
CREATE POLICY documents_isolation_policy ON app.documents
    FOR ALL
    TO PUBLIC
    USING (owner = current_user);

-- Create policy: Allow SELECT for all authenticated users
CREATE POLICY documents_select_policy ON app.documents
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Create policy: Users can only modify their own documents
CREATE POLICY documents_modify_policy ON app.documents
    FOR INSERT
    TO PUBLIC
    WITH CHECK (owner = current_user);

-- Grant privileges to roles
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE app.documents TO app_admin, app_user, app_readonly;
GRANT USAGE, SELECT ON SEQUENCE app.documents_id_seq TO app_admin, app_user;

-- Test RLS policies (in separate sessions):
-- SET ROLE app_user;
-- SELECT * FROM app.documents;  -- Returns all documents (SELECT policy)
-- INSERT INTO app.documents (title, content, owner) VALUES ('New Doc', 'Content', 'app_user');  -- Success
-- INSERT INTO app.documents (title, content, owner) VALUES ('Fake Doc', 'Content', 'app_admin');  -- Fails (owner mismatch)
-- RESET ROLE;

-- ============================================================================
-- 4. Multi-Tenant Isolation using Session Variables
-- ============================================================================

-- Create multi-tenant table
CREATE TABLE app.tenant_data (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert sample data for multiple tenants
INSERT INTO app.tenant_data (tenant_id, data) VALUES
    (1, 'Tenant 1 - Record A'),
    (1, 'Tenant 1 - Record B'),
    (2, 'Tenant 2 - Record A'),
    (2, 'Tenant 2 - Record B'),
    (3, 'Tenant 3 - Record A');

-- Enable RLS
ALTER TABLE app.tenant_data ENABLE ROW LEVEL SECURITY;

-- Create policy using session variable
CREATE POLICY tenant_isolation_policy ON app.tenant_data
    FOR ALL
    TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id')::INTEGER);

-- Grant privileges
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE app.tenant_data TO app_user;
GRANT USAGE, SELECT ON SEQUENCE app.tenant_data_id_seq TO app_user;

-- Test multi-tenant isolation:
-- SET app.current_tenant_id = '1';
-- SELECT * FROM app.tenant_data;  -- Returns only tenant 1 data
--
-- SET app.current_tenant_id = '2';
-- SELECT * FROM app.tenant_data;  -- Returns only tenant 2 data
--
-- RESET app.current_tenant_id;

-- Create tenant-specific roles
CREATE ROLE tenant_a_user WITH LOGIN PASSWORD 'tenant_a_password';
CREATE ROLE tenant_b_user WITH LOGIN PASSWORD 'tenant_b_password';

GRANT CONNECT ON DATABASE postgres TO tenant_a_user, tenant_b_user;
GRANT USAGE ON SCHEMA app TO tenant_a_user, tenant_b_user;
GRANT SELECT, INSERT, UPDATE ON TABLE app.tenant_data TO tenant_a_user, tenant_b_user;

-- Set default tenant_id for each role using ALTER ROLE
ALTER ROLE tenant_a_user SET app.current_tenant_id = '1';
ALTER ROLE tenant_b_user SET app.current_tenant_id = '2';

-- ============================================================================
-- 5. Audit Logging with Triggers
-- ============================================================================

-- Create audit log table
CREATE TABLE app.audit_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,  -- INSERT, UPDATE, DELETE
    record_id INTEGER,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT DEFAULT current_user,
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION app.audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        INSERT INTO app.audit_log (table_name, operation, record_id, old_data)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD)::jsonb);
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO app.audit_log (table_name, operation, record_id, old_data, new_data)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD)::jsonb, row_to_json(NEW)::jsonb);
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO app.audit_log (table_name, operation, record_id, new_data)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW)::jsonb);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Attach audit trigger to projects table
CREATE TRIGGER audit_projects_trigger
    AFTER INSERT OR UPDATE OR DELETE ON app.projects
    FOR EACH ROW EXECUTE FUNCTION app.audit_trigger_func();

-- Test audit logging
INSERT INTO app.projects (name, description, owner) VALUES
    ('Test Project', 'Test description', 'app_admin');

UPDATE app.projects SET description = 'Updated description' WHERE name = 'Test Project';

DELETE FROM app.projects WHERE name = 'Test Project';

-- View audit log
SELECT
    id,
    table_name,
    operation,
    record_id,
    changed_by,
    changed_at,
    old_data->>'name' AS old_name,
    new_data->>'name' AS new_name
FROM app.audit_log
ORDER BY changed_at DESC;

-- Grant read-only access to audit log
GRANT SELECT ON TABLE app.audit_log TO app_admin;

-- ============================================================================
-- 6. Encryption and Hashing with pgcrypto
-- ============================================================================

-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create table for sensitive data
CREATE TABLE app.sensitive_data (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,  -- Store hashed passwords
    email_encrypted BYTEA,  -- Store encrypted email
    api_key_encrypted BYTEA,  -- Store encrypted API key
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 6.1 Password Hashing
-- ============================================================================

-- Insert user with hashed password (using bcrypt)
INSERT INTO app.sensitive_data (username, password_hash)
VALUES ('alice', crypt('alice_secret_password', gen_salt('bf')));

INSERT INTO app.sensitive_data (username, password_hash)
VALUES ('bob', crypt('bob_secret_password', gen_salt('bf', 8)));  -- Cost factor 8

-- Verify password (returns true/false)
SELECT
    username,
    (password_hash = crypt('alice_secret_password', password_hash)) AS password_valid
FROM app.sensitive_data
WHERE username = 'alice';

-- Login function
CREATE OR REPLACE FUNCTION app.authenticate_user(
    p_username TEXT,
    p_password TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    v_password_hash TEXT;
BEGIN
    SELECT password_hash INTO v_password_hash
    FROM app.sensitive_data
    WHERE username = p_username;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    RETURN (v_password_hash = crypt(p_password, v_password_hash));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Test authentication
SELECT app.authenticate_user('alice', 'alice_secret_password');  -- true
SELECT app.authenticate_user('alice', 'wrong_password');  -- false

-- ============================================================================
-- 6.2 Symmetric Encryption (PGP)
-- ============================================================================

-- Encryption key (in production, store securely outside database)
DO $$
DECLARE
    encryption_key TEXT := 'my-secret-encryption-key-2024';
BEGIN
    -- Encrypt and store email
    UPDATE app.sensitive_data
    SET email_encrypted = pgp_sym_encrypt('alice@example.com', encryption_key)
    WHERE username = 'alice';

    UPDATE app.sensitive_data
    SET email_encrypted = pgp_sym_encrypt('bob@example.com', encryption_key)
    WHERE username = 'bob';
END $$;

-- Decrypt email (in application code, not directly in queries)
DO $$
DECLARE
    encryption_key TEXT := 'my-secret-encryption-key-2024';
BEGIN
    RAISE NOTICE 'Alice email: %', (
        SELECT pgp_sym_decrypt(email_encrypted, encryption_key)
        FROM app.sensitive_data
        WHERE username = 'alice'
    );
END $$;

-- ============================================================================
-- 6.3 Asymmetric Encryption (Public/Private Key)
-- ============================================================================

-- Generate key pair (in production, use external tools like GPG)
DO $$
DECLARE
    public_key TEXT := '-----BEGIN PGP PUBLIC KEY BLOCK-----
...
-----END PGP PUBLIC KEY BLOCK-----';
    private_key TEXT := '-----BEGIN PGP PRIVATE KEY BLOCK-----
...
-----END PGP PRIVATE KEY BLOCK-----';
    passphrase TEXT := 'key-passphrase';
BEGIN
    -- In practice, you would use real PGP keys here
    -- This is a placeholder to demonstrate the API

    -- Encrypt with public key
    -- UPDATE app.sensitive_data
    -- SET api_key_encrypted = pgp_pub_encrypt('sk-1234567890abcdef', dearmor(public_key))
    -- WHERE username = 'alice';

    -- Decrypt with private key
    -- SELECT pgp_pub_decrypt(api_key_encrypted, dearmor(private_key), passphrase)
    -- FROM app.sensitive_data
    -- WHERE username = 'alice';

    RAISE NOTICE 'Asymmetric encryption example (use real PGP keys in production)';
END $$;

-- ============================================================================
-- 6.4 Hashing (One-way)
-- ============================================================================

-- SHA-256 hash for data integrity checks
SELECT encode(digest('sensitive-data', 'sha256'), 'hex') AS sha256_hash;

-- HMAC for message authentication
SELECT encode(hmac('message', 'secret-key', 'sha256'), 'hex') AS hmac_sha256;

-- Example: API key hashing for secure storage
CREATE TABLE app.api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    key_hash TEXT NOT NULL,  -- Store hash, not plaintext
    key_prefix TEXT NOT NULL,  -- Store prefix for user identification (e.g., 'sk-1234...')
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Function to create API key
CREATE OR REPLACE FUNCTION app.create_api_key(p_user_id INTEGER)
RETURNS TEXT AS $$
DECLARE
    v_api_key TEXT;
    v_key_hash TEXT;
    v_key_prefix TEXT;
BEGIN
    -- Generate random API key
    v_api_key := 'sk-' || encode(gen_random_bytes(32), 'hex');

    -- Hash for storage
    v_key_hash := encode(digest(v_api_key, 'sha256'), 'hex');

    -- Store prefix for user reference
    v_key_prefix := substring(v_api_key from 1 for 10) || '...';

    -- Save to database
    INSERT INTO app.api_keys (user_id, key_hash, key_prefix)
    VALUES (p_user_id, v_key_hash, v_key_prefix);

    -- Return plaintext key (only time it's visible)
    RETURN v_api_key;
END;
$$ LANGUAGE plpgsql;

-- Function to validate API key
CREATE OR REPLACE FUNCTION app.validate_api_key(p_api_key TEXT)
RETURNS INTEGER AS $$
DECLARE
    v_user_id INTEGER;
BEGIN
    SELECT user_id INTO v_user_id
    FROM app.api_keys
    WHERE key_hash = encode(digest(p_api_key, 'sha256'), 'hex');

    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

-- Test API key management
DO $$
DECLARE
    new_key TEXT;
    user_id INTEGER;
BEGIN
    -- Create API key for user 1
    new_key := app.create_api_key(1);
    RAISE NOTICE 'Generated API key: %', new_key;

    -- Validate the key
    user_id := app.validate_api_key(new_key);
    RAISE NOTICE 'Key belongs to user: %', user_id;

    -- Try invalid key
    user_id := app.validate_api_key('sk-invalid-key');
    RAISE NOTICE 'Invalid key user: %', user_id;  -- NULL
END $$;

-- View stored API keys (hashes only)
SELECT id, user_id, key_prefix, created_at
FROM app.api_keys;

-- ============================================================================
-- 7. Security Best Practices Summary
-- ============================================================================

-- View all role privileges
SELECT
    r.rolname,
    r.rolsuper,
    r.rolinherit,
    r.rolcreaterole,
    r.rolcreatedb,
    r.rolcanlogin
FROM pg_roles r
WHERE r.rolname LIKE 'app_%' OR r.rolname LIKE 'tenant_%'
ORDER BY r.rolname;

-- View RLS policies
SELECT
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies
WHERE schemaname = 'app'
ORDER BY tablename, policyname;

-- ============================================================================
-- Security Checklist
-- ============================================================================
-- ✓ 1. Use roles with least privilege principle
-- ✓ 2. Grant specific privileges (SELECT, INSERT) instead of ALL
-- ✓ 3. Enable Row-Level Security for multi-tenant applications
-- ✓ 4. Use session variables for dynamic tenant isolation
-- ✓ 5. Implement audit logging for compliance
-- ✓ 6. Hash passwords with bcrypt (NEVER store plaintext)
-- ✓ 7. Encrypt sensitive data (email, API keys, PII)
-- ✓ 8. Use prepared statements to prevent SQL injection
-- ✓ 9. Regularly rotate encryption keys and credentials
-- ✓ 10. Monitor and review pg_stat_activity for suspicious queries
-- ✓ 11. Use SSL/TLS for client connections
-- ✓ 12. Restrict network access with pg_hba.conf
-- ✓ 13. Keep PostgreSQL updated with security patches
-- ✓ 14. Use connection pooling (pgBouncer) to limit connections
-- ✓ 15. Enable query logging for security audits
-- ============================================================================
