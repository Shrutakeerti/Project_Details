import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('D:\Sign_language_Detection_notRLT\sign_language_model.h5')


sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z'] 


def preprocess_frame(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    gray_resized = cv2.resize(gray, (28, 28))
    
    gray_resized = gray_resized.reshape(1, 28, 28, 1)
  
    gray_resized = gray_resized.astype('float32') / 255.0
    return gray_resized


prediction_buffer = []


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

   
    processed_frame = preprocess_frame(frame)

   
    prediction = model.predict(processed_frame)
    
    
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)

   
    if confidence > 0.75:  
        prediction_buffer.append(predicted_class[0])


    if len(prediction_buffer) > 10:  
        most_frequent_sign = max(set(prediction_buffer), key=prediction_buffer.count)
        predicted_sign = sign_labels[most_frequent_sign]
        prediction_buffer = [] 
    else:
        predicted_sign = "..."  

    
    cv2.putText(frame, f'Sign: {predicted_sign} | Confidence: {confidence:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Real-time Sign Language Prediction', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
