import sqlite3

conn = sqlite3.connect("company.db")
cursor = conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS quotations (
    quote_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    product_id INTEGER,
    quoted_price REAL,
    valid_until DATE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    product_id INTEGER,
    quantity INTEGER,
    order_date DATE,
    status TEXT COLLATE NOCASE, -- Case-insensitive comparison
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Seed Initial Data
INSERT OR IGNORE INTO products VALUES
(1, 'Industrial Pump X1', 'Machinery', 1200.0, 14),
(2, 'Steel Bearing Ring 50mm', 'Hardware', 45.5, 350),
(3, 'Control Valve V2', 'Plumbing', 310.0, 8);

INSERT OR IGNORE INTO quotations VALUES
(101, 'Nordic Logistics Oy', 1, 1150.0, '2026-10-30'),
(102, 'Satakunta Tech Ab', 2, 42.0, '2026-11-15');

INSERT OR IGNORE INTO orders VALUES
(501, 'Helsinki Engineering', 1, 2, '2026-09-01', 'Delivered'),
(502, 'Turku Marine Works', 3, 4, '2026-09-15', 'Processing');
""")

conn.commit()
conn.close()
print("Database company.db created successfully.")
