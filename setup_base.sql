-- Create the test1 database
CREATE DATABASE IF NOT EXISTS test1;

-- Use the test1 database
USE test1;

-- Create the table tab1
DROP TABLE IF EXISTS tab1;  -- This line ensures that if the table already exists, it's dropped first to avoid errors
CREATE TABLE tab1 (
    recordID INT(11) NOT NULL,
    name VARCHAR(30)
);

-- Insert values into tab1
INSERT INTO tab1 (recordID, name) VALUES
(1, 'Fred'),
(2, 'Freda');

-- SELECT to display all of the values and columns in table tab1
SELECT * FROM tab1;
