from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import os
from models import predict_image
from treatment_info import get_treatment_recommendations 
from database import db, User
from werkzeug.security import generate_password_hash, check_password_hash
from database import Prediction
import smtplib 
from email.message import EmailMessage

# Initialize app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load trained CNN model
#model, class_labels = load_cnn_model("best_model_overall.h5")
model_path="best_model_overall.h5"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['emailid']
        password = generate_password_hash(request.form['password'])

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, password=password,emailid=email)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file selected", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        pname=request.form['t1']
        pemail=request.form['t2']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence, grad_cam_filename, class_idx = predict_image(model_path, filepath)
        confidence=confidence*100
        
        # Get treatment recommendations
        treatment_info = get_treatment_recommendations(class_idx)
        new_prediction = Prediction(user_id=current_user.id,pname=pname,filename=filename,label=predicted_class,confidence=confidence)
        db.session.add(new_prediction)
        db.session.commit()
        msg = EmailMessage()
        email = pemail
        mes="Dear "+pname+ " For Your input image "+filename+" Predicted as "+predicted_class+" With confidence of "+str(confidence)
        msg.set_content(str(mes))
        msg['Subject'] = 'Alert'
        msg['From'] = "poisonousplants2024@gmail.com"
        msg['To'] = email
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login("poisonousplants2024@gmail.com", "wtfghdcknihmbaog")
        s.send_message(msg)
        s.quit()
        
        return render_template('result.html', 
                              filename=filename, 
                              label=predicted_class, 
                              confidence=confidence,
                              grad_cam_filename=grad_cam_filename,
                              treatment_info=treatment_info)

    return render_template('predict.html')
@app.route('/history')
@login_required
def history():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', predictions=user_predictions)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
