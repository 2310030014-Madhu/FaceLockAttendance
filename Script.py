from keras_facenet import FaceNet
import numpy as np
import cv2
from pymongo import MongoClient
embedder = FaceNet()  # Load FaceNet model
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_system"]
teachers_col = db["teachers"]
attendance_col = db["attendance"]
teachers = list(teachers_col.find())  # Find teachers without embeddings

# for teacher in teachers:
#     image_path = teacher["image_path"]
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     embeddings = embedder.embeddings([img_rgb])
#     if len(embeddings) > 0:
#         face_embedding = embeddings[0].tolist()

#         # Update the teacher record with the embedding
#         teachers_col.update_one(
#             {"teacher_id": teacher["teacher_id"]},
#             {"$set": {"face_embedding": face_embedding}}
#         )

# print("Updated all teachers with face embeddings.")

for t in teachers:
    print(t)# Print the face embedding of teacher with teacher_id 001