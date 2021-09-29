import dlib
import cv2
 
INPUT_FILE = './video/back2future.mp4'
OUTPUT_FILE = './video/test005.mp4'
start_frame = 0
end_frame = 450
 
reader = cv2.VideoCapture(INPUT_FILE)
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'I420')  
writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (width, height))# resolution

detector = dlib.get_frontal_face_detector() 
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

print(reader.isOpened())
# have_more_frame = True
c = 0
while True:
    ret, frame = reader.read()
    c += 1
    if c>= start_frame and c< 100:
        cv2.putText(frame, 'BGR TO GRAY', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(10)
        writer.write(frame)
        cv2.imshow("frame01", frame)
    if c>= 100 and c< 200:
        cv2.putText(frame, 'FLIP & CANNY', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flipframe = cv2.flip(frame, 1)
        edgeframe = cv2.Canny(flipframe, 32, 128)
        cv2.waitKey(10)
        writer.write(edgeframe)
        cv2.imshow("frame02", edgeframe)
    if c>= 200 and c< 270:
        cv2.putText(frame, 'FACE DETECTION', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        if ret :
            face_rects, scores, idx = detector.run(frame, 0, -.5)  # 偵測人臉

            for i, d in enumerate(face_rects):               # 取出所有偵測的結果
                x1 = d.left(); y1 = d.top(); x2 = d.right(); y2 = d.bottom()
                text = f'{scores[i]:.2f}, ({idx[i]:0.0f})'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA) # 以方框標示偵測的人臉
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,          # 標示分數
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.waitKey(10)
        writer.write(frame)
        cv2.imshow("frame03", frame)
    if c>= 300 and c< 450:
        cv2.putText(frame, 'KNN', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        fgframe = bs.apply(frame)
        cv2.waitKey(10)
        writer.write(fgframe)
        cv2.imshow("frame04", fgframe)
    if c>end_frame:
        print('complete!')
        break

writer.release()
reader.release()
cv2.destroyAllWindows()