DROP TABLE IF EXISTS entries;
CREATE TABLE entries (
    id INTEGER PRIMARY KEY autoincrement,
    title string NOT NULL,
    text string NOT NULL
);