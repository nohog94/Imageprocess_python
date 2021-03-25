from tkinter.filedialog import *
from tkinter.simpledialog import *
import cv2
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os.path
import requests
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from shapely.geometry import Point, Polygon

def malloc(row, col, value=0):
    retAry = [[[value for _ in range(col)] for _ in range(row)] for _ in range(3)]
    return retAry

def openImage():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG, BB
    fileName = askopenfilename(parent=window,
            filetypes=(('Color File', '*.png;*.jpg'),('All File','*.*')))
    inCvImage = cv2.imread(fileName)
    inH, inW = inCvImage.shape[:2]
    inImage = malloc(inH, inW)
    for i in range(inH):
        for k in range(inW):
            inImage[BB][i][k] = inCvImage.item(i, k, RR)
            inImage[GG][i][k] = inCvImage.item(i, k, GG)
            inImage[RR][i][k] = inCvImage.item(i, k, BB)
    outImage = copy.deepcopy(inImage)
    outH, outW = inH, inW
    displayImage()

def open_random_Image():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG, BB
    item = askstring("검색 열기", "키워드 입력")
    url = 'https://www.pexels.com/search/' + item

    options = Options()
    #options.headless = True
    driver = webdriver.Chrome(executable_path=r"C:\Users\kccistc\Downloads\chromedriver.exe", options=options)
    driver.get(url)
    driver.implicitly_wait(3)
    searchBox = driver.find_element_by_xpath("//img[@class='photo-item__img']")
    attribute = searchBox.get_attribute('srcset')
    url_addresss = attribute.split(',')[0]
    url_addresss = url_addresss[:-5]
    url_addresss = url_addresss + '&h=500'
    r = requests.get(url_addresss,stream=True)
    with open("temp.jpg", 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
    inCvImage = cv2.imread('temp.jpg')
    inH, inW = inCvImage.shape[:2]
    inImage = malloc(inH, inW)
    for i in range(inH):
        for k in range(inW):
            inImage[BB][i][k] = inCvImage.item(i, k, RR)
            inImage[GG][i][k] = inCvImage.item(i, k, GG)
            inImage[RR][i][k] = inCvImage.item(i, k, BB)
    outImage = copy.deepcopy(inImage)
    driver.close()
    outH, outW = inH, inW
    fileName = "temp.jpg"
    displayImage()

def displayImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG, BB
    ## 기존에 그림을 붙인적이 있으면, 게시판(canvas) 뜯어내기
    if canvas != None :
        canvas.destroy()
    window.geometry(str(outW) + "x" + str(outH))
    canvas = Canvas(window, height=outH, width=outW)
    paper = PhotoImage(height=outH, width=outW)
    canvas.create_image( (outW/2, outH/2), image=paper, state='normal')

    rgbString ="" # 전체 펜을 저장함
    for i in range(outH) :
        tmpString = "" # 각 1줄의 펜
        for k in range(outW) :
            rr = outImage[RR][i][k]
            gg = outImage[GG][i][k]
            bb = outImage[BB][i][k]
            tmpString += "#%02x%02x%02x " % (rr, gg, bb)  # 제일 뒤 공백 1칸
        rgbString += '{' + tmpString + '} ' # 제일 뒤 공백 1칸
    paper.put(rgbString)
    canvas.pack()
    status.configure(text= str(outW) + 'x' + str(outH) + '  ' + fileName)

def saveImage() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    if fileName == None :
        return
    saveCvImage = np.zeros((outH, outW, 3), np.uint8) # 데이터형식 지원
    for i in range(outH) :
        for k in range(outW) :
            tup = tuple(  (  [outImage[BB][i][k], outImage[GG][i][k], outImage[RR][i][k] ] ) )
            # --> ( ( [100, 77, 55] ) )
            saveCvImage[i,k] = tup
    saveFp = asksaveasfile(parent=window, mode='wb', defaultextension='.png',
                filetypes=(("Image Tyep", "*.png;*.jpg;*.bmp;*.tif"),("All File", "*.*")))
    if saveFp == '' or saveFp == None :
        return
    cv2.imwrite(saveFp.name, saveCvImage)
    print('Save is done')

def exit():
    global window
    window.destroy()

def guess_image():
    global fileName
    if os.path.isfile('mobilenet.h5') is False:
        mobile = tf.keras.applications.mobilenet.MobileNet()
        mobile.save('mobilenet.h5')
    else:
        mobile = load_model('mobilenet.h5', compile=False)
    img = image.load_img(fileName, target_size = (224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    print(f"This is {results[0][0][1]} of {results[0][0][2]}%")

##### 영상처리 함수 모음 #####
def grayscale_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    ## 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    ## *** 진짜 영상처리 알고리즘을 구현 ***
    for i in range(inH) :
       for k in range(inW) :
           hap = inImage[RR][i][k] + inImage[GG][i][k] + inImage[BB][i][k]
           outImage[RR][i][k] =  outImage[GG][i][k] =outImage[BB][i][k] = hap//3
    ##/////////////////////////////////////////////
    displayImage()

def add_image() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    ## 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    ## *** 진짜 영상처리 알고리즘을 구현 ***
    value = askinteger("밝게/어둡게", "값 입력", minvalue=-255, maxvalue=255)
    for rgb in range(RGB) :
        for i in range(inH) :
            for k in range(inW) :
                if (inImage[rgb][i][k] + value > 255) :
                    outImage[rgb][i][k] = 255
                elif (inImage[rgb][i][k] + value < 0 ) :
                    outImage[rgb][i][k] = 0
                else :
                    outImage[rgb][i][k] = inImage[rgb][i][k] + value
    ##/////////////////////////////////////////////
    displayImage()

def reverse():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    ## 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    for rgb in range(RGB):
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i][k] = 255 - inImage[rgb][i][k]
    ##/////////////////////////////////////////////
    displayImage()

def gamma():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    ## 중요! 출력이미지의 높이, 폭을 결정  --> 알고리즘에 영향
    outH = inH; outW = inW;
    outImage = malloc(outH, outW)
    gamma = askfloat("감마", "감마 값 입력", minvalue=-3.0, maxvalue=3.0)
    for rgb in range(RGB):
        for i in range(inH) :
            for k in range(inW) :
                outImage[rgb][i][k] = int(255*(inImage[rgb][i][k]/255)**gamma)
    ##/////////////////////////////////////////////
    displayImage()

def blur():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    outCvImage = cv2.blur(inCvImage, (3,3))
    outH, outW = inH, inW
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def edge_detect():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    gray = cv2.cvtColor(inCvImage, cv2.COLOR_BGR2GRAY)
    outCvImage = cv2.Canny(gray, 150, 175)
    outH, outW = inH, inW
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k)
            outImage[GG][i][k] = outCvImage.item(i, k)
            outImage[RR][i][k] = outCvImage.item(i, k)
    displayImage()

def sharpen():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    outCvImage = cv2.filter2D(inCvImage, -1, mask)
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def emboss():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    mask = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    outCvImage = cv2.filter2D(inCvImage, -1, mask)
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()


def move():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    x = askinteger("move", "x 값 입력", minvalue=-inW, maxvalue=inW)
    y = askinteger("move", "y 값 입력", minvalue=-inH, maxvalue=inH)
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (inW, inH)
    outCvImage = cv2.warpAffine(inCvImage, transMat, dimensions)
    outH, outW = inH, inW
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def rotate():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    angle = askinteger("rotate", "회전 각도 입력", minvalue=-360, maxvalue=360)
    outH, outW = inH, inW
    rotPoint = (inW//2, inH//2)
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (inW, inH)
    outCvImage = cv2.warpAffine(inCvImage, rotMat, dimensions)
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def resize():
    global outImage, outH, outW, inCvImage, outCvImage, RR, GG ,BB
    outH = askinteger("resize", "outH 값 입력", minvalue=1, maxvalue=1000)
    outW = askinteger("resize", "outW 값 입력", minvalue=1, maxvalue=1000)
    outCvImage = cv2.resize(inCvImage, (outW,outH), interpolation=cv2.INTER_CUBIC)
    outImage = malloc(outH, outW)
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def flip():
    global outImage, outH, outW, inCvImage, outCvImage, RR, GG ,BB
    value = askinteger("flip", "수평 filp은 1, 수직 flip은 0", minvalue=0, maxvalue=1)
    outCvImage = cv2.flip(inCvImage, value)
    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def face_detect():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    face_cascade = cv2.CascadeClassifier("haar_face.xml")
    gray = cv2.cvtColor(inCvImage, cv2.COLOR_RGBA2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.1,5)
    outCvImage = inCvImage[:]
    for (x,y,w,h) in face_rects:
        cv2.rectangle(outCvImage, (x,y), (x+w, y+h), (0,255,0), 3)

    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def face_recognition():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, inCvImage, outCvImage, RGB, RR, GG ,BB
    if (inImage == None) :
        return
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier("haar_face.xml")
    if os.path.isfile('face_trained.yml') is False:
        people = os.listdir("people")
        img_array = []
        labels = []
        for person in people:
            subdir = os.listdir('people\\' + person)
            label = people.index(person)
            for file in subdir:
                img = cv2.imread('people\\' + person + '\\' + file)
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                face_rects = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in face_rects:
                    face_roi = gray[y:y + h, x:x + w]
                img_array.append(face_roi)
                labels.append(label)
        img_array = np.array(img_array, dtype=object)
        labels = np.array(labels)
        face_recognizer.train(img_array, labels)
        face_recognizer.save('face_trained.yml')
    else:
        face_recognizer.read('face_trained.yml')

    gray = cv2.cvtColor(inCvImage, cv2.COLOR_RGBA2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.1,5)
    outCvImage = inCvImage[:]
    for (x,y,w,h) in face_rects:
        face_roi = gray[y:y + h, x:x + w]
    label, confidence = face_recognizer.predict(face_roi)
    guess = os.listdir("people")[label]
    print(f'{guess}입니다!')


    for i in range(outH):
        for k in range(outW):
            outImage[BB][i][k] = outCvImage.item(i, k, RR)
            outImage[GG][i][k] = outCvImage.item(i, k, GG)
            outImage[RR][i][k] = outCvImage.item(i, k, BB)
    displayImage()

def nega_image():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
    global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
    global sx, sy, ex, ey
    global xyList, rubbers, mousePress, saveX, saveY

    xyList = []  # 클릭하거나 Move한 점의 좌표 리스트 (= 폴리곤 좌표)
    rubbers = []  # 임시로 그려진 선들의 집합
    mousePress = False  # 클릭한 후에, 마우스를 Move 했는지 체크하기 위함.
    sx, sy, ex, ey = [0] * 4  # 임시 선을 그리기 위한 좌표
    saveX, saveY = 0, 0  # 첫 클릭한 좌표. 마지막 점과 다시 연결하기 위함

    def negaImage_click(event):
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        global xyList, rubbers, mousePress, saveX, saveY

        mousePress = True  # 클릭한 상태로 전환
        xyList.append((event.y, event.x))  # 첫 클릭점을 폴리곤 좌표 목록에 추가
        saveX = sx = event.x  # 첫점 및 임시선을 그리기 위한 시작점
        saveY = sy = event.y
        print(sx, sy)
    def negaImage_move(event):
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        global xyList, rubbers, mousePress, saveX, saveY

        if mousePress:  # 마우스를 클릭한 상태에서 움직인 것만 인정
            xyList.append((event.y, event.x))  # 마우스를 움직인 위치를 '계속' 폴리곤 좌표 목록에 추가
            ex = event.x  # 임시선을 그리기 위한 끝점
            ey = event.y
            rubber = canvas.create_line(sx, sy, ex, ey, fill="red")  # 임시선 그림
            rubbers.append(rubber)  # 임시선을 임시선 목록에 넣음
            sx = ex  # 다음 임시 선을 그리기 위해서 마지막 점을 다시 시작점으로...
            sy = ey

    def negaImage_drop(event):
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        global xyList, rubbers, mousePress, saveX, saveY

        mousePress = False  # 마우스를 드롭하면 체크
        ex = event.x
        ey = event.y
        rubber = canvas.create_line(ex, ey, saveX, saveY, fill="red")  # 마지막 점과 첫 점을 연결 (폴리곤 그림)
        rubbers.append(rubber)
        xyList.append((event.y, event.x))
        __negaImage()
        xyList = []  # 초기화
        rubbers = []  # 초기화
        canvas.unbind("<Button-1>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<Button-3>")

    def negaImage_rClick(event):
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        global xyList, rubbers, mousePress, saveX, saveY

        if len(xyList) == 0:
            xyList = None
        __negaImage()
        canvas.unbind("<Button-1>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<Button-3>")

    def __negaImage():
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW
        global fileName, cvInImage, cvOutImage, RGB, RR, GG, BB
        global sx, sy, ex, ey
        global xyList, rubbers, mousePress, saveX, saveY

        if fileName == None:
            return

        # 중요! 출력 영상의 크기 결정 (알고리즘에 따라서)
        outH = inH;
        outW = inW
        # 출력 영상 메모리 할당
        outImage = malloc(outH, outW)

        #### 진짜 영상처리 알고리즘 ####
        if xyList == None or len(xyList) < 3:  # 그냥 클릭만 했거나, 폴리곤이 아니면 전체를 처리
            for rgb in range(RGB):
                for i in range(inH):
                    for k in range(inW):
                        outImage[rgb][i][k] = 255 - inImage[rgb][i][k]

        else:  # 폴리곤이 3점 이상이면...
            poly = Polygon(xyList)  # 좌표 리스트를 폴리곤 객체로 만듬
            for rgb in range(RGB):
                for i in range(inH):
                    for k in range(inW):
                        # Point in Polygon Algorithm
                        pnt = Point(i, k)  # 현재 점 (전체 좌표를 하나씩 처리)
                        if (pnt.within(poly)):  # 현재 점이 폴리곤 안에 있으면 처리
                            outImage[rgb][i][k] = inImage[rgb][i][k]
                        else:
                            outImage[rgb][i][k] = 0
        displayImage()

    canvas.bind("<Button-1>",negaImage_click)
    #canvas.bind("<ButtonRelease-1>", negaImage_drop)
    canvas.bind("<Button-3>", negaImage_rClick)
    canvas.bind("<B1-Motion>", negaImage_move)




window, canvas, paper = None, None, None
inImage, outIamge = None, None
inH, inW, outH, outW = [0] * 4
filename = None
RGB, RR, GG, BB = 3, 0, 1, 2


window = Tk()
window.title("영상처리(파이썬) Beta1")
window.geometry('500x500')
window.resizable(width = False, height = False)
status = Label(window, text="이미지 정보: ", bd = 1, relief = SUNKEN, anchor = W)
status.pack(side = BOTTOM, fill= X)

mainMenu = Menu(window) # 메인메뉴
window.config(menu=mainMenu)

fileMenu = Menu(mainMenu)
mainMenu.add_cascade(label='파일', menu=fileMenu)
Image = Menu(fileMenu)
fileMenu.add_cascade(label='열기', menu=Image)
Image.add_command(label='로컬', command=openImage)
Image.add_command(label='검색', command=open_random_Image)
fileMenu.add_command(label='저장', command=saveImage)
fileMenu.add_separator()
fileMenu.add_command(label='종료', command=exit)

photoMenu1 = Menu(mainMenu)
mainMenu.add_cascade(label='화소 점 처리', menu=photoMenu1)
photoMenu1.add_command(label='밝게 하기', command=add_image)
photoMenu1.add_command(label='그레이스케일', command=grayscale_image)
photoMenu1.add_command(label='색상 반전', command=reverse)
photoMenu1.add_command(label='감마 변환', command=gamma)

photoMenu2 = Menu(mainMenu)
mainMenu.add_cascade(label='화소 영역 처리', menu=photoMenu2)
photoMenu2.add_command(label='Blur', command=blur)
photoMenu2.add_command(label='Edge Detection', command=edge_detect)
photoMenu2.add_command(label='Sharpen', command=sharpen)
photoMenu2.add_command(label='emboss', command=emboss)

photoMenu3 = Menu(mainMenu)
mainMenu.add_cascade(label='기하학적 처리', menu=photoMenu3)
photoMenu3.add_command(label='Move', command=move)
photoMenu3.add_command(label='Flip', command=flip)
photoMenu3.add_command(label='Rotate', command=rotate)
photoMenu3.add_command(label='resize', command=resize)

photoMenu4 = Menu(mainMenu)
mainMenu.add_cascade(label='추가 기능', menu=photoMenu4)
photoMenu4.add_command(label='Image Detection', command=guess_image)
photoMenu4.add_command(label='Face Detection', command=face_detect)
photoMenu4.add_command(label='Face Recognition', command=face_recognition)
photoMenu4.add_command(label='crop', command=nega_image)

window.mainloop()
