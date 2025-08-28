PRAGMA foreign_keys = ON;

-- ======================================
-- Tabele
-- ======================================
CREATE TABLE IF NOT EXISTS owners (
    owner_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    city TEXT
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id INTEGER PRIMARY KEY,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER CHECK(year >= 1900),
    list_price INTEGER,
    availability TEXT DEFAULT 'available', -- available | sold | reserved
    declared_type TEXT -- np. 'car'|'truck'|'motorcycle'
);

CREATE TABLE IF NOT EXISTS transaction_history (
    transaction_id INTEGER PRIMARY KEY,
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    buyer_id INTEGER REFERENCES owners(owner_id),
    seller_id INTEGER REFERENCES owners(owner_id),
    transaction_date TEXT NOT NULL, -- ISO 8601
    price INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tx_vehicle ON transaction_history(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_tx_date ON transaction_history(transaction_date);

CREATE TABLE IF NOT EXISTS vehicle_images (
    image_id INTEGER PRIMARY KEY,
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    image_url TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_img_vehicle ON vehicle_images(vehicle_id);

-- ======================================
-- Dane przykładowe
-- ======================================

INSERT INTO owners (owner_id, first_name, last_name, city) VALUES
(1,'Jan','Kowalski','Warszawa'),
(2,'Anna','Nowak','Kraków'),
(3,'Piotr','Zieliński','Gdańsk'),
(4,'Maria','Wiśniewska','Poznań');

INSERT INTO vehicles (vehicle_id, brand, model, year, list_price, availability, declared_type) VALUES
(1,'Toyota','Corolla',2018,45000,'sold','car'),
(2,'BMW','X5',2020,180000,'sold','car'),
(3,'MAN','TGS',2017,350000,'sold','truck'),
(4,'Honda','CBR600RR',2019,38000,'sold','motorcycle'),
(5,'Skoda','Octavia',2016,32000,'available','car');

INSERT INTO transaction_history (transaction_id, vehicle_id, buyer_id, seller_id, transaction_date, price) VALUES
(1,1,1,NULL,'2021-05-12',45000),
(2,2,2,NULL,'2022-01-08',180000),
(3,3,3,NULL,'2019-09-20',350000),
(4,1,4,1,'2023-02-15',40000),
(5,4,1,NULL,'2020-07-03',38000);

INSERT INTO vehicle_images (image_id, vehicle_id, image_url) VALUES
(1,1,'images/toyota_corolla_2018.jpg'),
(2,2,'images/bmw_x5_2020.jpg'),
(3,3,'images/man_tgs_2017.jpg'),
(4,4,'images/honda_cbr600rr_2016.jpg'),
(5,5,'images/skoda_octavia_2016.jpg');
