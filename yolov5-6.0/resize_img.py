import os
import cv2
import numpy as np

def get_paths(home_dir):
    images = []
    for home, dirs, files in os.walk(home_dir):
        for filename in files:
            if os.path.splitext(filename)[-1] in [".jpg", ".bmp", ".png", ".jpeg"]:
                path = os.path.join(home, filename)
                images.append(path)               
    return images

if __name__=="__main__":
    root_dir = r"E:\Study\yolov5\Project\pinyu\T0001"
    dst_dir = r"E:\Study\yolov5\Project\pinyu\T0003"
    dst_size = [1024, 1024, 3]
    padding = 60
    for path in get_paths(root_dir):
        print(path)
        try:
            new_img = np.zeros(dst_size, dtype=np.uint8)
            name = path.split(root_dir+os.sep)[-1]
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            src_h, src_w, src_c = img.shape
            roi_img = img
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.inRange(gray, np.array([50]), np.array([255]))
            # # cv2.imshow("a", gray)
            # # cv2.waitKey(0)
            # cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = sorted(cnts, key=lambda x:cv2.contourArea(x), reverse=True)
            # x, y, w, h = cv2.boundingRect(cnts[0])
            # x = x-padding if x-padding > 0 else 0
            # y = y-padding if y-padding > 0 else 0
            # x2 = x+w+2*padding if x+w+2*padding < src_w else src_w
            # y2 = y+h+2*padding if y+h+2*padding < src_h else src_h
            # roi_img = img[y:y2, x:x2]
            h,w,c = roi_img.shape
            h_scale = h/dst_size[0]
            w_scale = w/dst_size[1]
            scale = max([h_scale, w_scale])
            dst_h = int(h / scale)
            dst_w = int(w / scale)
            resized = cv2.resize(roi_img, [dst_w, dst_h], interpolation=cv2.INTER_AREA)
            new_img[:dst_h, :dst_w] = resized
            new_path = os.path.join(dst_dir, name)
            new_dir = os.path.split(new_path)[0]
            ext = os.path.splitext(new_path)[-1]
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            cv2.imencode(ext, new_img)[1].tofile(new_path)
            cv2.imwrite(new_path, new_img)
        except:
            print(path)
