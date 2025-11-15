-- VoxBank Database Schema

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    account_type VARCHAR(50) NOT NULL, -- 'savings', 'current', 'salary'
    balance DECIMAL(15, 2) DEFAULT 0.00,
    currency VARCHAR(10) DEFAULT 'INR',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    from_account_id VARCHAR(50),
    to_account_id VARCHAR(50),
    amount DECIMAL(15, 2) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL, -- 'transfer', 'deposit', 'withdrawal'
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (to_account_id) REFERENCES accounts(account_id)
);

-- Loans table
CREATE TABLE IF NOT EXISTS loans (
    loan_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    loan_type VARCHAR(50) NOT NULL, -- 'personal', 'home', 'car'
    principal_amount DECIMAL(15, 2) NOT NULL,
    interest_rate DECIMAL(5, 2) NOT NULL,
    emi_amount DECIMAL(15, 2) NOT NULL,
    remaining_balance DECIMAL(15, 2) NOT NULL,
    next_emi_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Reminders table
CREATE TABLE IF NOT EXISTS reminders (
    reminder_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    reminder_type VARCHAR(50) NOT NULL, -- 'payment', 'bill', 'custom'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    due_date DATE NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed', 'cancelled'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Voiceprints table (for biometric authentication)
CREATE TABLE IF NOT EXISTS voiceprints (
    user_id VARCHAR(50) PRIMARY KEY,
    embedding BLOB, -- Store voiceprint embedding
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- OTP table
CREATE TABLE IF NOT EXISTS otp_codes (
    otp_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    otp_code VARCHAR(6) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Indexes for performance
CREATE INDEX idx_accounts_user_id ON accounts(user_id);
CREATE INDEX idx_transactions_from_account ON transactions(from_account_id);
CREATE INDEX idx_transactions_to_account ON transactions(to_account_id);
CREATE INDEX idx_transactions_created_at ON transactions(created_at);
CREATE INDEX idx_loans_user_id ON loans(user_id);
CREATE INDEX idx_reminders_user_id ON reminders(user_id);
CREATE INDEX idx_reminders_due_date ON reminders(due_date);

