// backend/middleware/auth.js
const jwt = require('jsonwebtoken');
const User = require('../models/User');

module.exports = async function(req, res, next){
  const header = req.headers['authorization'];
  if(!header) return res.status(401).send({error: 'Missing auth token'});
  const token = header.split(' ')[1];
  try{
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = await User.findById(decoded.id).select('-passwordHash');
    if(!req.user) return res.status(401).send({error: 'User not found'});
    next();
  }catch(err){
    res.status(401).send({error: 'Invalid token'});
  }
};
