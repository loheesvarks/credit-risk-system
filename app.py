"""
Flask API for Credit Risk Scoring System
Provides endpoints for risk prediction, threshold optimization, and loss simulation
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
import os
from auth import (
    register_user, login_user, logout_user, verify_token,
    require_auth, save_prediction, get_user_predictions
)

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load model artifacts
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_PATH, 'model.joblib'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.joblib'))

with open(os.path.join(MODEL_PATH, 'model_config.json'), 'r') as f:
    config = json.load(f)

FEATURE_NAMES = config['feature_names']

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_auc': config['roc_auc']})

# ============== AUTH ROUTES ==============

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    result = register_user(name, email, password)
    if result['success']:
        return jsonify(result), 201
    return jsonify(result), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    result = login_user(email, password)
    if result['success']:
        return jsonify(result), 200
    return jsonify(result), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    logout_user(token)
    return jsonify({'success': True})

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    return jsonify({'user': request.user})

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Handle Google OAuth login/register"""
    data = request.json
    email = data.get('email', '').strip().lower()
    name = data.get('name', '')
    google_id = data.get('googleId', '')
    photo = data.get('photo', '')
    
    if not email or not google_id:
        return jsonify({'error': 'Invalid Google data'}), 400
    
    conn = __import__('auth').get_db()
    try:
        # Check if user exists
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        
        if user:
            # User exists, create session
            token = __import__('auth').generate_token()
            expires_at = __import__('datetime').datetime.now() + __import__('datetime').timedelta(days=7)
            
            conn.execute(
                'INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)',
                (user['id'], token, expires_at)
            )
            conn.execute(
                'UPDATE users SET last_login = ? WHERE id = ?',
                (__import__('datetime').datetime.now(), user['id'])
            )
            conn.commit()
            
            return jsonify({
                'success': True,
                'token': token,
                'user': {
                    'id': user['id'],
                    'name': user['name'],
                    'email': user['email'],
                    'role': user['role']
                }
            })
        else:
            # Create new user
            conn.execute(
                'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                (name, email, f'google:{google_id}')
            )
            conn.commit()
            
            user = conn.execute(
                'SELECT * FROM users WHERE email = ?', (email,)
            ).fetchone()
            
            token = __import__('auth').generate_token()
            expires_at = __import__('datetime').datetime.now() + __import__('datetime').timedelta(days=7)
            
            conn.execute(
                'INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)',
                (user['id'], token, expires_at)
            )
            conn.commit()
            
            return jsonify({
                'success': True,
                'token': token,
                'user': {
                    'id': user['id'],
                    'name': user['name'],
                    'email': user['email'],
                    'role': user['role']
                }
            }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/api/auth/history', methods=['GET'])
@require_auth
def get_history():
    history = get_user_predictions(request.user['id'])
    return jsonify({'history': history})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict default probability for a single applicant"""
    try:
        data = request.json
        features = [data.get(f, 0) for f in FEATURE_NAMES]
        features_scaled = scaler.transform([features])
        
        probability = float(model.predict_proba(features_scaled)[0, 1])
        threshold = data.get('threshold', config['optimal_threshold'])
        
        # Risk categorization
        if probability < 0.1:
            risk_category = 'Very Low'
        elif probability < 0.25:
            risk_category = 'Low'
        elif probability < 0.5:
            risk_category = 'Medium'
        elif probability < 0.75:
            risk_category = 'High'
        else:
            risk_category = 'Very High'
        
        # Expected loss calculation
        loan_amount = data.get('loan_amount', 1000000)  # ₹10 Lakh default
        lgd = data.get('lgd', config['default_lgd'])
        expected_loss = probability * lgd * loan_amount
        
        return jsonify({
            'probability': round(probability, 4),
            'risk_category': risk_category,
            'decision': 'Reject' if probability >= threshold else 'Approve',
            'threshold_used': threshold,
            'expected_loss': round(expected_loss, 2),
            'loan_amount': loan_amount
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict for multiple applicants"""
    try:
        data = request.json
        applicants = data.get('applicants', [])
        threshold = data.get('threshold', config['optimal_threshold'])
        
        results = []
        for applicant in applicants:
            features = [applicant.get(f, 0) for f in FEATURE_NAMES]
            features_scaled = scaler.transform([features])
            prob = float(model.predict_proba(features_scaled)[0, 1])
            
            results.append({
                'id': applicant.get('id', len(results)),
                'probability': round(prob, 4),
                'decision': 'Reject' if prob >= threshold else 'Approve'
            })
        
        return jsonify({'results': results, 'threshold': threshold})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/threshold_analysis', methods=['POST'])
def threshold_analysis():
    """Analyze different threshold impacts on business metrics"""
    try:
        data = request.json
        probabilities = data.get('probabilities', [])
        loan_amounts = data.get('loan_amounts', [10000] * len(probabilities))
        actual_defaults = data.get('actual_defaults', None)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        analysis = []
        
        for thresh in thresholds:
            decisions = ['Reject' if p >= thresh else 'Approve' for p in probabilities]
            approved = sum(1 for d in decisions if d == 'Approve')
            rejected = sum(1 for d in decisions if d == 'Reject')
            approval_rate = approved / len(decisions) if decisions else 0
            
            # Expected loss for approved loans
            approved_probs = [p for p, d in zip(probabilities, decisions) if d == 'Approve']
            approved_amounts = [a for a, d in zip(loan_amounts, decisions) if d == 'Approve']
            
            expected_loss = sum(p * 0.45 * a for p, a in zip(approved_probs, approved_amounts))
            total_approved_amount = sum(approved_amounts)
            
            analysis.append({
                'threshold': round(float(thresh), 2),
                'approved': approved,
                'rejected': rejected,
                'approval_rate': round(approval_rate, 3),
                'expected_loss': round(expected_loss, 2),
                'total_approved_amount': round(total_approved_amount, 2)
            })
        
        return jsonify({'analysis': analysis, 'optimal_threshold': config['optimal_threshold']})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/loss_simulation', methods=['POST'])
def loss_simulation():
    """Simulate business losses under different scenarios"""
    try:
        data = request.json
        num_applicants = data.get('num_applicants', 1000)
        avg_loan_amount = data.get('avg_loan_amount', 1200000)  # ₹12 Lakh default
        threshold = data.get('threshold', config['optimal_threshold'])
        lgd = data.get('lgd', config['default_lgd'])
        interest_rate = data.get('interest_rate', 0.10)  # 10% for India
        
        # Generate synthetic portfolio
        np.random.seed(data.get('seed', 42))
        probabilities = np.random.beta(2, 8, num_applicants)
        loan_amounts = np.random.lognormal(np.log(avg_loan_amount), 0.5, num_applicants)
        
        # Apply threshold
        approved_mask = probabilities < threshold
        approved_probs = probabilities[approved_mask]
        approved_amounts = loan_amounts[approved_mask]
        
        # Simulate defaults
        defaults = np.random.random(len(approved_probs)) < approved_probs
        
        # Calculate metrics
        total_approved = len(approved_amounts)
        total_approved_amount = float(np.sum(approved_amounts))
        num_defaults = int(np.sum(defaults))
        actual_loss = float(np.sum(approved_amounts[defaults] * lgd))
        expected_loss = float(np.sum(approved_probs * lgd * approved_amounts))
        
        # Revenue from non-defaults
        revenue = float(np.sum(approved_amounts[~defaults] * interest_rate))
        net_profit = revenue - actual_loss
        
        return jsonify({
            'total_applicants': num_applicants,
            'total_approved': total_approved,
            'approval_rate': round(total_approved / num_applicants, 3),
            'total_approved_amount': round(total_approved_amount, 2),
            'num_defaults': num_defaults,
            'default_rate': round(num_defaults / total_approved, 4) if total_approved > 0 else 0,
            'actual_loss': round(actual_loss, 2),
            'expected_loss': round(expected_loss, 2),
            'revenue': round(revenue, 2),
            'net_profit': round(net_profit, 2),
            'threshold_used': threshold
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get model configuration"""
    return jsonify(config)

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance rankings"""
    importance = config['feature_importance']
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return jsonify({'feature_importance': sorted_importance})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
