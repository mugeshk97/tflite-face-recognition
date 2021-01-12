import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
    return cropped_face

def process_input(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = cv2.resize(img,(128,128))
    img = np.expand_dims(img, 2)
    img = np.expand_dims(img, 0)
    return img

interpreter = tflite.Interpreter(model_path='facenet.tflite')
interpreter.allocate_tensors()

video_capture = cv2.VideoCapture(0)
while True:
    cls = ['Manoj', 'Mugesh']
    ret, frame = video_capture.read()
    if ret:
        face = face_extractor(frame)
    else:
        print('assertion error')
        video_capture.release()
        cv2.destroyAllWindows()
        break
    if face is not None:
        img = process_input(face)
        input_data = (np.float32(img))
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        s = cls[np.argmax(output_data)]
        cv2.putText(frame, s, (0, 185), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
        cv2.imshow('Frame', frame)
    else:
        print('face not found')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()