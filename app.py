import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify,Response,session,flash
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import timedelta,datetime
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from functools import wraps
from keras_facenet import FaceNet
from PIL import Image
import io



app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

embedder = FaceNet()  # Load FaceNet model

app.config["SESSION_PERMANENT"] = False  # Session will expire on browser close
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30) 

uri = "mongodb+srv://darrenRing:DarrenRing@123@cluster0.vpxfx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["attendance_system"]
teachers_col = db["teachers"]
attendance_col = db["attendance"]

app.secret_key = "batman"

admin_col = db["admins"]


def save_teacher(name, teacher_id, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate FaceNet embedding
    embeddings = embedder.embeddings([img_rgb])
    if len(embeddings) == 0:
        return False  # No face detected

    face_embedding = embeddings[0].tolist()  # Convert to list for MongoDB storage

    # Update teacher entry in the database
    teachers_col.update_one(
        {"teacher_id": teacher_id},
        {"$set": {
            "name": name,
            "image_path": image_path,
            "face_embedding": face_embedding  # Store FaceNet embedding
        }},
        upsert=True
    )
    return True

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        admin = admin_col.find_one({"username": username})
        print(admin,check_password_hash(admin["password"], password))
        if admin and check_password_hash(admin["password"], password):
            session["admin"] = username  
            return redirect(url_for("dashboard"))

        return render_template("index.html", error="Invalid credentials")

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")  # Admin dashboard page





def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("Checking if admin is logged in...")  # Debug print
        if "admin" not in session:
            print("Admin not logged in! Redirecting to login.")  # Debug print
            flash("You must be logged in as admin to access this page.", "danger")
            return redirect(url_for("index"))  # Redirect to login
        print("Admin is logged in! Access granted.")  # Debug print
        return f(*args, **kwargs)
    return decorated_function


@app.route("/capture_image", methods=["POST"])
@admin_required
def capture_image():
    camera = cv2.VideoCapture(0)  # Open webcam only for registration
    ret, frame = camera.read()
    camera.release()  # Release immediately after capturing

    if not ret:
        return "Failed to capture image."

    image_path = os.path.join("static/uploads", "register.jpg")
    cv2.imwrite(image_path, frame)

    return image_path

@app.route("/mark_attendance")
@admin_required
def mark_attendance_page():
    return render_template("mark_attendance.html")

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/capture_attendance", methods=["POST"])
def capture_attendance():
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    camera.release()

    if not success:
        return "Failed to capture image."

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_embedding = embedder.embeddings([img_rgb])[0]

    # Load all stored teacher embeddings
    teachers = list(teachers_col.find({}))
    teacher_embeddings = {t["teacher_id"]: (t["name"], np.array(t["face_embedding"])) for t in teachers}

    # Compare embeddings using Euclidean distance

    for teacher_id, (teacher_name,stored_embedding) in teacher_embeddings.items():
        distance = np.linalg.norm(stored_embedding - captured_embedding)
        if distance < 0.9:  # Threshold for match
            today_date = datetime.now().strftime("%Y-%m-%d")
            month = datetime.now().strftime("%Y-%m")

            attendance_col.update_one(
                {"teacher_id": teacher_id, "date": today_date},
                {"$set": {
                    "name": teacher_name,
                    "month": month,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }},
                upsert=True
            )
            return f"Attendance marked for {teacher_id}"

    return "No match found.CHECK"



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.pop("admin", None) 
    return redirect(url_for("index"))  


@app.route("/register", methods=["GET", "POST"])
@admin_required
def register():
    if request.method == "POST":
        name = request.form["name"]
        teacher_id = request.form["teacher_id"]
        image_data = request.form["image_data"]

        if not image_data:
            return "No image captured!"

        # Decode Base64 image
        image_data = image_data.split(",")[1]  # Remove the Base64 header
        image_binary = base64.b64decode(image_data)
        image_path = f"static/uploads/{teacher_id}.jpg"

        # Save image to disk
        with open(image_path, "wb") as f:
            f.write(image_binary)

        # Convert to NumPy array for FaceNet processing
        image = Image.open(io.BytesIO(image_binary)).convert("RGB")
        image = np.array(image)

        # Detect and encode face
        faces = embedder.extract(image, threshold=0.95)  # Returns list of detected faces with embeddings
        if not faces:
            return "No face detected in the captured image!"

        face_embedding = faces[0]["embedding"].tolist()  # Get first detected face's embedding

        # Store teacher details + face embedding in MongoDB
        teachers_col.insert_one({
            "name": name,
            "teacher_id": teacher_id,
            "image_path": image_path,
            "face_embedding": face_embedding
        })

        return render_template("confirm.html", name=name)

    return render_template("register.html")



@app.route("/teachers")
@admin_required
def teachers():
    selected_month = request.args.get("month", "")
    teacher_id_filter = request.args.get("teacher_id", "")
    name_filter = request.args.get("name", "").strip().lower()

    query = {}
    if selected_month:
        query["month"] = selected_month
    if teacher_id_filter:
        query["teacher_id"] = teacher_id_filter
    if name_filter:
        query["name"] = {"$regex": name_filter, "$options": "i"}  # Case-insensitive name search

    teachers = list(teachers_col.find({}))  # Get all registered teachers
    attendance_logs = list(attendance_col.find(query))  # Apply filters

    # Create a mapping of teacher_id -> photo for easy lookup
    teacher_photos = {teacher["teacher_id"]: teacher["image_path"] for teacher in teachers}

    # Add photo info to attendance logs
    for log in attendance_logs:
        log["photo"] = teacher_photos.get(log["teacher_id"], "default.jpg")
        

    # Count attendance for filtered results
    attendance_counts = attendance_col.aggregate([
        {"$match": query},
        {"$group": {"_id": "$teacher_id", "count": {"$sum": 1}}}
    ])
    attendance_count_map = len(attendance_logs)
    return render_template(
        "teachers.html",
        teachers=teachers,
        attendance_logs=attendance_logs,
        selected_month=selected_month,
        attendance_count_map=attendance_count_map
    )
 
 
@app.route("/home")
@admin_required
def home():
    return render_template("dashboard.html") 
    
@app.route("/manage_teachers")
@admin_required
def manage_teachers():
    teachers = list(teachers_col.find({}))
    return render_template("manage_teachers.html", teachers=teachers)

@app.route("/edit_teacher/<teacher_id>", methods=["GET", "POST"])
def edit_teacher(teacher_id):
    teacher = teachers_col.find_one({"_id": ObjectId(teacher_id)})

    if request.method == "POST":
        name = request.form["name"]
        new_teacher_id = request.form["teacher_id"]
        updated_data = {"name": name, "teacher_id": new_teacher_id}

        # Handle captured photo (base64)
        photo_data = request.form.get("photo_data")
        if photo_data:
            photo_data = photo_data.split(",")[1]  # Remove base64 header
            photo_bytes = base64.b64decode(photo_data)
            filename = f"{new_teacher_id}.jpg"
            photo_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            with open(photo_path, "wb") as f:
                f.write(photo_bytes)

            updated_data["image_path"] = photo_path  # Update image path in MongoDB

            # Extract new FaceNet embedding
            face_encoder = FaceNet()
            img = cv2.imread(photo_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = face_encoder.embeddings([img_rgb])[0]

            if embedding is not None:
                updated_data["face_embedding"] = embedding.tolist()  # Convert to list for MongoDB storage
            else:
                return "Error: No face detected in the uploaded image!"

        # Update teacher details in MongoDB
        teachers_col.update_one({"_id": ObjectId(teacher["_id"])}, {"$set": updated_data})

        # Update attendance records to reflect the new teacher_id and name
        attendance_col.update_many(
            {"teacher_id": teacher["teacher_id"]},  # Match old teacher_id
            {"$set": {"teacher_id": new_teacher_id, "name": name}}
        )

        return redirect(url_for("manage_teachers"))

    return render_template("edit_teacher.html", teacher=teacher)



@app.route("/delete_teacher/<teacher_id>", methods=["POST"])    
def delete_teacher(teacher_id):
    # Delete teacher from the teachers collection
    teachers_col.delete_one({"_id": ObjectId(teacher_id)})
    
    # Delete all attendance logs of the corresponding teacher
    attendance_col.delete_many({"teacher_id": teacher_id})
    
    return redirect(url_for("manage_teachers"))


if __name__ == "__main__":
    app.run(debug=True)
