import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

characteristic1 = []
characteristic2 = []
angle1 = 0
angle2 = 0


def loadModel():
    global characteristic1, characteristic2
    for i in range(8):
        name = str(str(i)+".jpg")
        
        getParams(name)
        
        characteristic1.append(int(np.degrees(angle1)))
        characteristic2.append(int(np.degrees(angle2)))
            
        print("wczytano obraz:", str(i)+".jpg")
                    

def getParams(title):
    global angle1, angle2
    img= cv2.imread(title)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
        
    for face in faces:
        
        landmarks=predictor(gray, face)
        
        params = [48, 57, 51, 62, 54]
        axisX = []
        axisY = []
        
        for i in range(5):
            axisX.append(landmarks.part(int(params[i])).x)
            axisY.append(landmarks.part(int(params[i])).y)
          
            
        left=np.array([axisX[0],axisY[0]])
        bottom=np.array([axisX[1],axisY[1]])
        top=np.array([axisX[2],axisY[2]])
        middle=np.array([axisX[3],axisY[3]])
        right=np.array([axisX[4],axisY[4]])
            
    
        ba = left - bottom
        bc = top - bottom
            
        ad = left - middle
        db = right - middle

        cosine_angle1 = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle1 = np.arccos(cosine_angle1)
            
        cosine_angle2 = np.dot(ad, db) / (np.linalg.norm(ad) * np.linalg.norm(db))
        angle2 = np.arccos(cosine_angle2)
        

loadModel()
print(characteristic1, characteristic2)

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['#FF0000','#00AAFF', '#00FF00'])


features=list(zip(characteristic1, characteristic2))
y_train= [0, 0, 1, 1, 0, 1, 0, 0] # 1- uśmiech, 0- brak brak emocji
X = np.array(features)
y = np.array(y_train)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

nazwa = ' '
img= cv2.imread(nazwa)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
faces = detector(gray)
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

for face in faces:
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()
    cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0),3)
    # predykcja keypointow
    landmarks=predictor(gray, face)
    for i in range(67):
        x=landmarks.part(i).x
        y=landmarks.part(i).y
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    getParams(nazwa)
    model = KNeighborsClassifier(n_neighbors=2)
    
    model.fit(features,y_train)
    
    predicted= model.predict([[np.degrees(angle1), np.degrees(angle2)]])
    if predicted == 1:
        cv2.putText(img, "Usmiech", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (125, 246, 55), 3)
    else:
        cv2.putText(img, "Brak emocji", (200, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (125, 246, 55), 3)

cv2.imshow("Face lanndmarks",img)
k = cv2.waitKey(0)
if k==27:  # escape
    cv2.destroyAllWindows()
	
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.scatter(np.degrees(angle1),np.degrees(angle2),0,'orange')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Mapa klasyfikacji dla k=2")
plt.show()