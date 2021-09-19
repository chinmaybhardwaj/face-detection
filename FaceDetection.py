import cv2


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')

    
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # Loop through each face and create a rectangle
    for x, y, width, height in faces:
        roi_color = frame[y:y + height, x:x + width]
        roi_gray = gray[y:y + height, x:x + width]
        
        # BGR
        color = (255, 0, 0) 
        stroke = 2       
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, stroke)
        
        #  Detect Eyes   
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ewidth, eheight in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ewidth, ey + eheight), (0, 255, 0), 2)
            
        # Profile Face
        profile = profile_cascade.detectMultiScale(roi_gray)
        for px, py, pwidth, pheight in profile:
            cv2.rectangle(roi_color, (px, py), (px + pwidth, py + pheight), (0, 255, 0), 2)
        
        # Show frame with rectangle around face
        cv2.imshow('frame', frame)
        
    # Stop capturing video and Exit 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

        
