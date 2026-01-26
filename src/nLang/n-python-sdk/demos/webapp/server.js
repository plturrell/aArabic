#!/usr/bin/env node
const express = require('express');
const path = require('path');

const app = express();
const PORT = 8084;

// Disable caching
app.use((req, res, next) => {
    res.header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
    res.header('Pragma', 'no-cache');
    res.header('Expires', '0');
    next();
});

// Serve static files from webapp directory
app.use(express.static(__dirname));

// Fallback to index.html for SPA
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`⚛ Intelligence SDK webapp serving at http://localhost:${PORT}`);
});