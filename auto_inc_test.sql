-- Create the test1 database
CREATE DATABASE IF NOT EXISTS test1;

-- Use the test1 database
USE test1;

-- Create the table tab1 with auto-increment capability for recordID
DROP TABLE IF EXISTS tab1;
CREATE TABLE tab1 (
    recordID INT(11) AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(30) DEFAULT NULL
);

-- Insert values into tab1
INSERT INTO tab1 (name) VALUES
('Fred'),
('Freda');

-- SELECT to display all of the values and columns in table tab1
SELECT * FROM tab1;
