-- Create the customers table
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    address TEXT
);

-- Create the accounts table
CREATE TABLE accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    account_type TEXT CHECK (account_type IN ('savings', 'checking', 'credit')),
    balance REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create the transactions table
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER,
    transaction_type TEXT CHECK (transaction_type IN ('deposit', 'withdrawal', 'transfer')),
    amount REAL NOT NULL,
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Insert sample customers
INSERT INTO customers (name, email, phone, address) VALUES
    ('Alice Johnson', 'alice.j@example.com', '123-456-7890', '123 Elm St.'),
    ('Bob Smith', 'bob.s@example.com', '234-567-8901', '456 Oak St.'),
    ('Charlie Davis', 'charlie.d@example.com', '345-678-9012', '789 Pine St.');

-- Insert sample accounts
INSERT INTO accounts (customer_id, account_type, balance) VALUES
    (1, 'savings', 5000.00),
    (1, 'checking', 1500.00),
    (2, 'checking', 2500.00),
    (3, 'savings', 7000.00);

-- Insert sample transactions
INSERT INTO transactions (account_id, transaction_type, amount) VALUES
    (1, 'deposit', 200.00),
    (1, 'withdrawal', 100.00),
    (2, 'deposit', 500.00),
    (3, 'withdrawal', 200.00),
    (4, 'deposit', 1000.00);
