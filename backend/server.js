// backend/server.js
require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const authRoutes = require('./routes/auth');
const patientRoutes = require('./routes/patients');
const predictionRoutes = require('./routes/predictions');
const modelRoutes = require('./routes/models');

const app = express();
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());

app.use('/api/auth', authRoutes);
app.use('/api/patients', patientRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/models', modelRoutes);

const PORT = process.env.PORT || 4000;
const MONGO_URI = process.env.MONGO_URI;

mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
.then(()=> {
  console.log('Mongo connected');
  app.listen(PORT, ()=> console.log('Backend listening on', PORT));
}).catch(err => {
  console.error('Mongo connection error', err);
});
