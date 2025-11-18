-- ============================================================================
-- VoxBank Database Setup Script (100% CLEAN - NO PLACEHOLDERS)
-- ============================================================================

DROP DATABASE IF EXISTS voxbank;
CREATE DATABASE voxbank;

\c voxbank;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- TABLES
-- ============================================================================

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    address TEXT,
    date_of_birth DATE,
    preferred_language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'closed')),
    kyc_status VARCHAR(20) DEFAULT 'pending' CHECK (kyc_status IN ('pending', 'verified', 'rejected')),
    passphrase VARCHAR(255),    -- demo-only login passphrase (hash in real systems)
    audio_embedding JSONB,           -- stored voice embedding for future voice auth
    last_active TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_phone ON users(phone_number);

CREATE TABLE accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_number VARCHAR(20) UNIQUE NOT NULL,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    account_type VARCHAR(20) NOT NULL CHECK (account_type IN ('savings', 'checking', 'fixed_deposit', 'business')),
    currency VARCHAR(3) DEFAULT 'USD' CHECK (currency IN ('USD', 'EUR', 'GBP', 'INR', 'CAD')),
    balance DECIMAL(15, 2) DEFAULT 0.00 CHECK (balance >= 0),
    available_balance DECIMAL(15, 2) DEFAULT 0.00,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'frozen', 'closed', 'dormant')),
    interest_rate DECIMAL(5, 2) DEFAULT 0.00,
    overdraft_limit DECIMAL(15, 2) DEFAULT 0.00,
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_accounts_user_id ON accounts(user_id);
CREATE INDEX idx_accounts_account_number ON accounts(account_number);
CREATE INDEX idx_accounts_status ON accounts(status);
CREATE INDEX idx_accounts_type ON accounts(account_type);

CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_reference VARCHAR(50) NOT NULL,
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
    from_account_id UUID REFERENCES accounts(account_id) ON DELETE RESTRICT,
    to_account_id UUID REFERENCES accounts(account_id) ON DELETE RESTRICT,
    entry_type VARCHAR(10) NOT NULL CHECK (entry_type IN ('debit', 'credit')),
    transaction_type VARCHAR(30) NOT NULL CHECK (transaction_type IN 
        ('transfer', 'deposit', 'withdrawal', 'payment', 'refund', 'fee', 'interest')),
    amount DECIMAL(15, 2) NOT NULL CHECK (amount > 0),
    currency VARCHAR(3) NOT NULL,
    fee DECIMAL(10, 2) DEFAULT 0.00,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN 
        ('pending', 'processing', 'completed', 'failed', 'cancelled', 'reversed')),
    description TEXT,
    metadata JSONB,
    balance_after DECIMAL(15, 2),
    initiated_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_transactions_account_id ON transactions(account_id, created_at DESC);
CREATE INDEX idx_transactions_reference ON transactions(transaction_reference);
CREATE INDEX idx_transactions_from_account ON transactions(from_account_id);
CREATE INDEX idx_transactions_to_account ON transactions(to_account_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);

CREATE TABLE beneficiaries (
    beneficiary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    beneficiary_account_number VARCHAR(20) NOT NULL,
    beneficiary_name VARCHAR(255) NOT NULL,
    nickname VARCHAR(100),
    bank_name VARCHAR(255),
    bank_code VARCHAR(20),
    is_internal BOOLEAN DEFAULT true,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, beneficiary_account_number)
);

CREATE INDEX idx_beneficiaries_user_id ON beneficiaries(user_id);
CREATE INDEX idx_beneficiaries_status ON beneficiaries(status);
CREATE INDEX idx_beneficiaries_account_number ON beneficiaries(beneficiary_account_number);

CREATE TABLE cards (
    card_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
    card_number VARCHAR(19) UNIQUE NOT NULL,
    card_type VARCHAR(20) NOT NULL CHECK (card_type IN ('debit', 'credit', 'prepaid')),
    cardholder_name VARCHAR(255) NOT NULL,
    expiry_month INTEGER NOT NULL CHECK (expiry_month BETWEEN 1 AND 12),
    expiry_year INTEGER NOT NULL CHECK (expiry_year >= EXTRACT(YEAR FROM CURRENT_DATE)),
    cvv VARCHAR(4) NOT NULL,
    pin_hash VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'blocked', 'expired', 'lost', 'stolen')),
    daily_limit DECIMAL(15, 2) DEFAULT 5000.00,
    monthly_limit DECIMAL(15, 2) DEFAULT 50000.00,
    contactless_enabled BOOLEAN DEFAULT true,
    online_transactions_enabled BOOLEAN DEFAULT true,
    international_enabled BOOLEAN DEFAULT false,
    issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cards_account_id ON cards(account_id);
CREATE INDEX idx_cards_card_number ON cards(card_number);
CREATE INDEX idx_cards_status ON cards(status);
CREATE INDEX idx_cards_expiry ON cards(expiry_year, expiry_month);

CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(token_hash);
CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE VIEW vw_account_summary AS
SELECT 
    a.account_id, a.account_number, a.account_type, a.balance, a.currency, a.status AS account_status,
    u.user_id, u.username, u.full_name, u.email, u.phone_number
FROM accounts a
JOIN users u ON a.user_id = u.user_id;

CREATE VIEW vw_transaction_history AS
SELECT 
    t.transaction_id, t.transaction_reference, t.account_id, a.account_number, a.user_id, u.full_name AS account_holder,
    t.entry_type, t.transaction_type, t.amount, t.currency, t.fee, t.status, t.description, t.balance_after,
    t.from_account_id, fa.account_number AS from_account_number, t.to_account_id, ta.account_number AS to_account_number,
    t.created_at, t.completed_at
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
JOIN users u ON a.user_id = u.user_id
LEFT JOIN accounts fa ON t.from_account_id = fa.account_id
LEFT JOIN accounts ta ON t.to_account_id = ta.account_id;

CREATE VIEW vw_card_details AS
SELECT 
    c.card_id, c.card_number, c.card_type, c.cardholder_name, c.expiry_month, c.expiry_year,
    c.status AS card_status, c.daily_limit, c.monthly_limit,
    a.account_id, a.account_number, a.account_type, a.balance, u.user_id, u.full_name, u.email
FROM cards c
JOIN accounts a ON c.account_id = a.account_id
JOIN users u ON a.user_id = u.user_id;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

CREATE OR REPLACE FUNCTION get_account_balance(p_account_id UUID)
RETURNS DECIMAL(15,2) AS $$
DECLARE v_balance DECIMAL(15,2);
BEGIN
    SELECT balance INTO v_balance FROM accounts WHERE account_id = p_account_id;
    RETURN COALESCE(v_balance, 0);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_transaction_count(p_account_id UUID)
RETURNS INTEGER AS $$
DECLARE v_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_count FROM transactions WHERE account_id = p_account_id;
    RETURN COALESCE(v_count, 0);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_beneficiaries_updated_at BEFORE UPDATE ON beneficiaries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cards_updated_at BEFORE UPDATE ON cards FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();



-- ============================================================================
-- VoxBank Sample Data Script
-- ============================================================================

\c voxbank;

-- ============================================================================
-- SAMPLE USERS
-- ============================================================================

INSERT INTO users (user_id, username, email, password_hash, full_name, phone_number, address, date_of_birth, preferred_language, status, kyc_status, last_active) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'john_doe', 'john.doe@email.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'John Doe', '+1-555-0101', '123 Main St, New York, NY 10001', '1985-03-15', 'en', 'active', 'verified', NOW() - INTERVAL '2 hours'),
('550e8400-e29b-41d4-a716-446655440002', 'jane_smith', 'jane.smith@email.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'Jane Smith', '+1-555-0102', '456 Oak Ave, Los Angeles, CA 90210', '1990-07-22', 'en', 'active', 'verified', NOW() - INTERVAL '1 day'),
('550e8400-e29b-41d4-a716-446655440003', 'mike_wilson', 'mike.wilson@email.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'Mike Wilson', '+1-555-0103', '789 Pine Rd, Chicago, IL 60601', '1988-11-08', 'en', 'active', 'pending', NOW() - INTERVAL '3 days'),
('550e8400-e29b-41d4-a716-446655440004', 'sarah_brown', 'sarah.brown@email.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'Sarah Brown', '+1-555-0104', '321 Elm St, Miami, FL 33101', '1992-05-18', 'en', 'active', 'verified', NOW() - INTERVAL '5 hours'),
('550e8400-e29b-41d4-a716-446655440005', 'david_jones', 'david.jones@email.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'David Jones', '+1-555-0105', '654 Maple Dr, Seattle, WA 98101', '1983-12-03', 'en', 'inactive', 'verified', NOW() - INTERVAL '1 week');

-- ============================================================================
-- SAMPLE ACCOUNTS
-- ============================================================================

INSERT INTO accounts (account_id, account_number, user_id, account_type, currency, balance, available_balance, status, interest_rate, overdraft_limit) VALUES
('660e8400-e29b-41d4-a716-446655440001', 'ACC001001', '550e8400-e29b-41d4-a716-446655440001', 'checking', 'USD', 15750.50, 15750.50, 'active', 0.25, 1000.00),
('660e8400-e29b-41d4-a716-446655440002', 'ACC001002', '550e8400-e29b-41d4-a716-446655440001', 'savings', 'USD', 45000.00, 45000.00, 'active', 2.50, 0.00),
('660e8400-e29b-41d4-a716-446655440003', 'ACC002001', '550e8400-e29b-41d4-a716-446655440002', 'checking', 'USD', 8920.75, 8920.75, 'active', 0.25, 500.00),
('660e8400-e29b-41d4-a716-446655440004', 'ACC002002', '550e8400-e29b-41d4-a716-446655440002', 'savings', 'USD', 22500.00, 22500.00, 'active', 2.50, 0.00),
('660e8400-e29b-41d4-a716-446655440005', 'ACC003001', '550e8400-e29b-41d4-a716-446655440003', 'checking', 'USD', 3250.25, 3250.25, 'active', 0.25, 250.00),
('660e8400-e29b-41d4-a716-446655440006', 'ACC004001', '550e8400-e29b-41d4-a716-446655440004', 'checking', 'USD', 12800.00, 12800.00, 'active', 0.25, 750.00),
('660e8400-e29b-41d4-a716-446655440007', 'ACC004002', '550e8400-e29b-41d4-a716-446655440004', 'business', 'USD', 75000.00, 75000.00, 'active', 1.75, 5000.00),
('660e8400-e29b-41d4-a716-446655440008', 'ACC005001', '550e8400-e29b-41d4-a716-446655440005', 'savings', 'USD', 18750.00, 18750.00, 'frozen', 2.50, 0.00),
('660e8400-e29b-41d4-a716-446655440009', 'ACC005002', '550e8400-e29b-41d4-a716-446655440005', 'fixed_deposit', 'USD', 50000.00, 0.00, 'active', 4.25, 0.00);

-- ============================================================================
-- SAMPLE TRANSACTIONS
-- ============================================================================

INSERT INTO transactions (transaction_id, transaction_reference, account_id, from_account_id, to_account_id, entry_type, transaction_type, amount, currency, fee, status, description, balance_after, initiated_by, completed_at) VALUES
('770e8400-e29b-41d4-a716-446655440001', 'TXN001', '660e8400-e29b-41d4-a716-446655440001', NULL, NULL, 'credit', 'deposit', 5000.00, 'USD', 0.00, 'completed', 'Initial deposit', 5000.00, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '30 days'),
('770e8400-e29b-41d4-a716-446655440002', 'TXN002', '660e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440003', 'debit', 'transfer', 250.00, 'USD', 2.50, 'completed', 'Transfer to Jane', 4747.50, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '25 days'),
('770e8400-e29b-41d4-a716-446655440003', 'TXN003', '660e8400-e29b-41d4-a716-446655440003', '660e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440003', 'credit', 'transfer', 250.00, 'USD', 0.00, 'completed', 'Transfer from John', 250.00, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '25 days'),
('770e8400-e29b-41d4-a716-446655440004', 'TXN004', '660e8400-e29b-41d4-a716-446655440001', NULL, NULL, 'debit', 'withdrawal', 500.00, 'USD', 3.00, 'completed', 'ATM withdrawal', 4244.50, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '20 days'),
('770e8400-e29b-41d4-a716-446655440005', 'TXN005', '660e8400-e29b-41d4-a716-446655440002', NULL, NULL, 'credit', 'deposit', 45000.00, 'USD', 0.00, 'completed', 'Savings account opening', 45000.00, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '28 days'),
('770e8400-e29b-41d4-a716-446655440006', 'TXN006', '660e8400-e29b-41d4-a716-446655440003', NULL, NULL, 'credit', 'deposit', 10000.00, 'USD', 0.00, 'completed', 'Salary deposit', 10250.00, '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '15 days'),
('770e8400-e29b-41d4-a716-446655440007', 'TXN007', '660e8400-e29b-41d4-a716-446655440003', NULL, NULL, 'debit', 'payment', 1329.25, 'USD', 0.00, 'completed', 'Rent payment', 8920.75, '550e8400-e29b-41d4-a716-446655440002', NOW() - INTERVAL '10 days'),
('770e8400-e29b-41d4-a716-446655440008', 'TXN008', '660e8400-e29b-41d4-a716-446655440001', NULL, NULL, 'credit', 'deposit', 12000.00, 'USD', 0.00, 'completed', 'Bonus payment', 16244.50, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '8 days'),
('770e8400-e29b-41d4-a716-446655440009', 'TXN009', '660e8400-e29b-41d4-a716-446655440001', NULL, NULL, 'debit', 'fee', 494.00, 'USD', 0.00, 'completed', 'Monthly maintenance fee', 15750.50, '550e8400-e29b-41d4-a716-446655440001', NOW() - INTERVAL '5 days'),
('770e8400-e29b-41d4-a716-446655440010', 'TXN010', '660e8400-e29b-41d4-a716-446655440006', NULL, NULL, 'credit', 'deposit', 15000.00, 'USD', 0.00, 'completed', 'Business revenue', 15000.00, '550e8400-e29b-41d4-a716-446655440004', NOW() - INTERVAL '12 days'),
('770e8400-e29b-41d4-a716-446655440011', 'TXN011', '660e8400-e29b-41d4-a716-446655440007', NULL, NULL, 'credit', 'deposit', 75000.00, 'USD', 0.00, 'completed', 'Business loan', 75000.00, '550e8400-e29b-41d4-a716-446655440004', NOW() - INTERVAL '18 days'),
('770e8400-e29b-41d4-a716-446655440012', 'TXN012', '660e8400-e29b-41d4-a716-446655440006', NULL, NULL, 'debit', 'payment', 2200.00, 'USD', 0.00, 'completed', 'Supplier payment', 12800.00, '550e8400-e29b-41d4-a716-446655440004', NOW() - INTERVAL '3 days'),
('770e8400-e29b-41d4-a716-446655440013', 'TXN013', '660e8400-e29b-41d4-a716-446655440005', NULL, NULL, 'credit', 'deposit', 3250.25, 'USD', 0.00, 'completed', 'Freelance payment', 3250.25, '550e8400-e29b-41d4-a716-446655440003', NOW() - INTERVAL '6 days');

-- ============================================================================
-- SAMPLE BENEFICIARIES
-- ============================================================================

INSERT INTO beneficiaries (beneficiary_id, user_id, beneficiary_account_number, beneficiary_name, nickname, bank_name, bank_code, is_internal, status) VALUES
('880e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'ACC002001', 'Jane Smith', 'Jane', 'VoxBank', 'VB001', true, 'active'),
('880e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440001', 'EXT123456', 'Robert Johnson', 'Bob', 'Chase Bank', 'CH001', false, 'active'),
('880e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440002', 'ACC001001', 'John Doe', 'John', 'VoxBank', 'VB001', true, 'active'),
('880e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440002', 'EXT789012', 'Emily Davis', 'Em', 'Bank of America', 'BOA001', false, 'active'),
('880e8400-e29b-41d4-a716-446655440005', '550e8400-e29b-41d4-a716-446655440003', 'ACC004001', 'Sarah Brown', 'Sarah', 'VoxBank', 'VB001', true, 'active'),
('880e8400-e29b-41d4-a716-446655440006', '550e8400-e29b-41d4-a716-446655440004', 'ACC001001', 'John Doe', 'John D', 'VoxBank', 'VB001', true, 'active'),
('880e8400-e29b-41d4-a716-446655440007', '550e8400-e29b-41d4-a716-446655440004', 'EXT345678', 'Michael Chen', 'Mike C', 'Wells Fargo', 'WF001', false, 'active'),
('880e8400-e29b-41d4-a716-446655440008', '550e8400-e29b-41d4-a716-446655440005', 'ACC002001', 'Jane Smith', 'Jane S', 'VoxBank', 'VB001', true, 'active');

-- ============================================================================
-- SAMPLE CARDS
-- ============================================================================

INSERT INTO cards (card_id, account_id, card_number, card_type, cardholder_name, expiry_month, expiry_year, cvv, pin_hash, status, daily_limit, monthly_limit, contactless_enabled, online_transactions_enabled, international_enabled, last_used_at) VALUES
('990e8400-e29b-41d4-a716-446655440001', '660e8400-e29b-41d4-a716-446655440001', '4532-1234-5678-9012', 'debit', 'JOHN DOE', 12, 2027, '123', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 2500.00, 25000.00, true, true, false, NOW() - INTERVAL '2 days'),
('990e8400-e29b-41d4-a716-446655440002', '660e8400-e29b-41d4-a716-446655440002', '4532-2345-6789-0123', 'debit', 'JOHN DOE', 8, 2028, '456', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 1000.00, 10000.00, true, true, false, NOW() - INTERVAL '1 week'),
('990e8400-e29b-41d4-a716-446655440003', '660e8400-e29b-41d4-a716-446655440003', '4532-3456-7890-1234', 'debit', 'JANE SMITH', 6, 2027, '789', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 3000.00, 30000.00, true, true, true, NOW() - INTERVAL '4 days'),
('990e8400-e29b-41d4-a716-446655440004', '660e8400-e29b-41d4-a716-446655440004', '4532-4567-8901-2345', 'debit', 'JANE SMITH', 3, 2029, '012', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 1500.00, 15000.00, true, false, false, NOW() - INTERVAL '10 days'),
('990e8400-e29b-41d4-a716-446655440005', '660e8400-e29b-41d4-a716-446655440005', '4532-5678-9012-3456', 'debit', 'MIKE WILSON', 11, 2026, '345', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 2000.00, 20000.00, false, true, false, NOW() - INTERVAL '1 day'),
('990e8400-e29b-41d4-a716-446655440006', '660e8400-e29b-41d4-a716-446655440006', '4532-6789-0123-4567', 'debit', 'SARAH BROWN', 9, 2028, '678', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 4000.00, 40000.00, true, true, true, NOW() - INTERVAL '3 hours'),
('990e8400-e29b-41d4-a716-446655440007', '660e8400-e29b-41d4-a716-446655440007', '4532-7890-1234-5678', 'credit', 'SARAH BROWN', 5, 2027, '901', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'active', 10000.00, 100000.00, true, true, true, NOW() - INTERVAL '5 days'),
('990e8400-e29b-41d4-a716-446655440008', '660e8400-e29b-41d4-a716-446655440008', '4532-8901-2345-6789', 'debit', 'DAVID JONES', 2, 2026, '234', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', 'blocked', 5000.00, 50000.00, true, true, false, NOW() - INTERVAL '2 weeks');

-- ============================================================================
-- SAMPLE AUDIT LOGS
-- ============================================================================

INSERT INTO audit_logs (log_id, user_id, action, entity_type, entity_id, old_values, new_values, ip_address, user_agent) VALUES
('aa0e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'LOGIN', 'user', '550e8400-e29b-41d4-a716-446655440001', NULL, '{"login_time": "2024-11-15T10:30:00Z"}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
('aa0e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440001', 'TRANSFER', 'transaction', '770e8400-e29b-41d4-a716-446655440002', NULL, '{"amount": 250.00, "to_account": "ACC002001"}', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
('aa0e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440002', 'LOGIN', 'user', '550e8400-e29b-41d4-a716-446655440002', NULL, '{"login_time": "2024-11-14T15:45:00Z"}', '10.0.0.50', 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)'),
('aa0e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440004', 'CARD_BLOCK', 'card', '990e8400-e29b-41d4-a716-446655440008', '{"status": "active"}', '{"status": "blocked", "reason": "suspicious_activity"}', '172.16.0.25', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'),
('aa0e8400-e29b-41d4-a716-446655440005', '550e8400-e29b-41d4-a716-446655440001', 'WITHDRAWAL', 'transaction', '770e8400-e29b-41d4-a716-446655440004', NULL, '{"amount": 500.00, "location": "ATM_001"}', '192.168.1.100', 'ATM Terminal');

-- ============================================================================
-- SAMPLE USER SESSIONS
-- ============================================================================

INSERT INTO user_sessions (session_id, user_id, token_hash, ip_address, user_agent, expires_at) VALUES
('bb0e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', NOW() + INTERVAL '24 hours'),
('bb0e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', '10.0.0.50', 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)', NOW() + INTERVAL '12 hours'),
('bb0e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440004', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/VjPoyNdO2', '172.16.0.25', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', NOW() + INTERVAL '8 hours');

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================

SELECT 'Sample data inserted successfully!' AS status;

SELECT 
    'users' AS table_name, COUNT(*) AS record_count FROM users
UNION ALL SELECT 'accounts', COUNT(*) FROM accounts
UNION ALL SELECT 'transactions', COUNT(*) FROM transactions
UNION ALL SELECT 'beneficiaries', COUNT(*) FROM beneficiaries
UNION ALL SELECT 'cards', COUNT(*) FROM cards
UNION ALL SELECT 'audit_logs', COUNT(*) FROM audit_logs
UNION ALL SELECT 'user_sessions', COUNT(*) FROM user_sessions
ORDER BY table_name;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'users' AS table_name, COUNT(*) AS record_count FROM users
UNION ALL SELECT 'accounts', COUNT(*) FROM accounts
UNION ALL SELECT 'transactions', COUNT(*) FROM transactions
UNION ALL SELECT 'beneficiaries', COUNT(*) FROM beneficiaries
UNION ALL SELECT 'cards', COUNT(*) FROM cards
UNION ALL SELECT 'audit_logs', COUNT(*) FROM audit_logs
UNION ALL SELECT 'user_sessions', COUNT(*) FROM user_sessions;

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'VoxBank database created successfully!';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Users: 5 | Accounts: 9 | Transactions: 13';
    RAISE NOTICE 'Beneficiaries: 8 | Cards: 8';
    RAISE NOTICE '============================================';
END $$;