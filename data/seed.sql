-- Seed data for VoxBank

-- Insert test users
INSERT INTO users (user_id, name, email, phone) VALUES
('USER001', 'Ramesh Kumar', 'ramesh.kumar@example.com', '+919876543210'),
('USER002', 'Priya Sharma', 'priya.sharma@example.com', '+919876543211'),
('USER003', 'Amit Patel', 'amit.patel@example.com', '+919876543212')
ON CONFLICT (user_id) DO NOTHING;

-- Insert test accounts
INSERT INTO accounts (account_id, user_id, account_type, balance, currency) VALUES
('ACC001', 'USER001', 'savings', 50000.00, 'INR'),
('ACC002', 'USER001', 'current', 25000.00, 'INR'),
('ACC003', 'USER002', 'savings', 75000.00, 'INR'),
('ACC004', 'USER003', 'savings', 30000.00, 'INR')
ON CONFLICT (account_id) DO NOTHING;

-- Insert sample transactions
INSERT INTO transactions (transaction_id, from_account_id, to_account_id, amount, transaction_type, status, description, created_at) VALUES
('TXN001', 'ACC001', 'ACC003', 5000.00, 'transfer', 'completed', 'Payment for services', NOW() - INTERVAL '5 days'),
('TXN002', 'ACC003', 'ACC001', 2000.00, 'transfer', 'completed', 'Refund', NOW() - INTERVAL '3 days'),
('TXN003', 'ACC001', NULL, 10000.00, 'deposit', 'completed', 'Salary credit', NOW() - INTERVAL '1 day')
ON CONFLICT (transaction_id) DO NOTHING;

-- Insert sample loans
INSERT INTO loans (loan_id, user_id, loan_type, principal_amount, interest_rate, emi_amount, remaining_balance, next_emi_date) VALUES
('LOAN001', 'USER001', 'personal', 500000.00, 12.5, 15000.00, 450000.00, CURRENT_DATE + INTERVAL '15 days'),
('LOAN002', 'USER002', 'home', 5000000.00, 8.5, 50000.00, 4800000.00, CURRENT_DATE + INTERVAL '20 days')
ON CONFLICT (loan_id) DO NOTHING;

-- Insert sample reminders
INSERT INTO reminders (reminder_id, user_id, reminder_type, title, description, due_date, status) VALUES
('REM001', 'USER001', 'payment', 'Loan EMI Payment', 'Pay EMI of ₹15,000 for loan LOAN001', CURRENT_DATE + INTERVAL '15 days', 'pending'),
('REM002', 'USER002', 'bill', 'Credit Card Bill', 'Pay credit card bill of ₹25,000', CURRENT_DATE + INTERVAL '10 days', 'pending')
ON CONFLICT (reminder_id) DO NOTHING;

