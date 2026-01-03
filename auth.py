"""
Authentication module with SQLite database
"""
import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with users table"""
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            role TEXT DEFAULT 'user'
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_data TEXT NOT NULL,
            probability REAL NOT NULL,
            decision TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()
    print("✓ Database initialized")

def hash_password(password):
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${hash_obj.hex()}"

def verify_password(password, password_hash):
    """Verify password against hash"""
    try:
        salt, hash_val = password_hash.split('$')
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hash_obj.hex() == hash_val
    except:
        return False

def generate_token():
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def register_user(name, email, password):
    """Register a new user"""
    conn = get_db()
    try:
        password_hash = hash_password(password)
        conn.execute(
            'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
            (name, email, password_hash)
        )
        conn.commit()
        return {'success': True, 'message': 'User registered successfully'}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'Email already registered'}
    finally:
        conn.close()

def login_user(email, password):
    """Authenticate user and create session"""
    conn = get_db()
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE email = ? AND is_active = 1',
            (email,)
        ).fetchone()
        
        if not user or not verify_password(password, user['password_hash']):
            return {'success': False, 'error': 'Invalid email or password'}
        
        # Create session token
        token = generate_token()
        expires_at = datetime.now() + timedelta(days=7)
        
        conn.execute(
            'INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)',
            (user['id'], token, expires_at)
        )
        
        # Update last login
        conn.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now(), user['id'])
        )
        conn.commit()
        
        return {
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role']
            }
        }
    finally:
        conn.close()

def verify_token(token):
    """Verify session token and return user"""
    conn = get_db()
    try:
        session = conn.execute(
            '''SELECT s.*, u.id as user_id, u.name, u.email, u.role 
               FROM sessions s 
               JOIN users u ON s.user_id = u.id 
               WHERE s.token = ? AND s.expires_at > ?''',
            (token, datetime.now())
        ).fetchone()
        
        if session:
            return {
                'id': session['user_id'],
                'name': session['name'],
                'email': session['email'],
                'role': session['role']
            }
        return None
    finally:
        conn.close()

def logout_user(token):
    """Invalidate session token"""
    conn = get_db()
    try:
        conn.execute('DELETE FROM sessions WHERE token = ?', (token,))
        conn.commit()
        return {'success': True}
    finally:
        conn.close()

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        user = verify_token(token)
        if not user:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.user = user
        return f(*args, **kwargs)
    return decorated

def save_prediction(user_id, input_data, probability, decision):
    """Save prediction to history"""
    conn = get_db()
    try:
        import json
        conn.execute(
            'INSERT INTO prediction_history (user_id, input_data, probability, decision) VALUES (?, ?, ?, ?)',
            (user_id, json.dumps(input_data), probability, decision)
        )
        conn.commit()
    finally:
        conn.close()

def get_user_predictions(user_id, limit=50):
    """Get user's prediction history"""
    conn = get_db()
    try:
        import json
        rows = conn.execute(
            'SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
            (user_id, limit)
        ).fetchall()
        return [{
            'id': r['id'],
            'input_data': json.loads(r['input_data']),
            'probability': r['probability'],
            'decision': r['decision'],
            'created_at': r['created_at']
        } for r in rows]
    finally:
        conn.close()

# Initialize database on import
init_db()
