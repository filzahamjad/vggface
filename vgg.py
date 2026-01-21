from deepface import DeepFace

# It automatically detects faces, aligns them, and compares them
result = DeepFace.verify(
    img1_path = 'face3/ID_1.jpg',
    img2_path = '/Users/filzahamjad/Desktop/sites/vggface/face3/Selfie_1.jpg',
    model_name = "VGG-Face", # Or "Facenet", "Facenet512", "OpenFace", "DeepFace"
    detector_backend = "opencv" # Options: 'retinaface', 'mtcnn', 'mediapipe'
)

print(f"Are they the same? {result['verified']}")
print(f"Distance: {result['distance']}")


