<?php
/**
 * Reads the file .env and creates the conection to the database.
 * This way the sensible data such as user and password is not
 * exposed directly into the code that is uploaded in GitHub
 */
 
// --- Read the .env file and load its values ---
function loadEnv($path) {
    if (!file_exists($path)) {
        die(".env file not found");
    }
 
    $rows = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
 
    foreach ($rows as $row) {
        $row = trim($row);
 
        list($key, $value) = explode("=", $row, 2);
        putenv(trim($key) . "=" . trim($value));
    }
}
 
loadEnv(__DIR__ . "/.env");
 
// --- Create the conection using .env values ---
function connectDB() {
    $connection = new mysqli(
        getenv("DB_HOST"),
        getenv("DB_USER"),
        getenv("DB_PASSWORD"),
        getenv("DB_NAME")
    );
 
    return $connection;
}
 