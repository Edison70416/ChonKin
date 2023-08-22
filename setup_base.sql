CREATE DATABASE IF NOT EXISTS test1;
USE test1;

-- Create the table tab1
DROP TABLE IF EXISTS tab1; 
CREATE TABLE tab1 (
    recordID INT(11) NOT NULL,
    name VARCHAR(30)
);

INSERT INTO tab1 (recordID, name) VALUES
(1, 'Fred'),
(2, 'Freda');
SELECT * FROM tab1;
