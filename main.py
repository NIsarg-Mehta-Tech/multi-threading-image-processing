import cv2 as cv
import threading
from queue import Queue
import matplotlib.pyplot as plt

img = cv.imread('images/nature.jpg')

if img is None:
    print('Could not read image')
    exit(0)

(h,w) = img.shape[:2]
dpi = 100
figsize = (w / dpi, h / dpi)
centerX, centerY = (w // 2), (h // 2)

q = Queue()
lock = threading.Lock()

threads = []

topLeft  = img[0:centerY, 0:centerX]
topRight  = img[0:centerY, centerX:w]
bottomLeft  = img[centerY:h, 0:centerX]
bottomRight  = img[centerY:h, centerX:w]

parts = [
    {"name" : "topLeft", "coords": (0, 0), "slice": topLeft}, 
    {"name": "topRight", "coords" : (0, centerX), "slice": topRight},
    {"name": "bottomLeft", "coords" : (centerY, 0), "slice": bottomLeft}, 
    {"name": "bottomRight", "coords": (centerY, centerX), "slice": bottomRight}
    ]

plt.figure(figsize=figsize, dpi=dpi)
plt.subplot(141)
plt.axis('off')
plt.imshow(cv.cvtColor(topLeft, cv.COLOR_BGR2RGB))
plt.title('Top Left')
plt.subplot(142)
plt.axis('off')
plt.imshow(cv.cvtColor(topRight, cv.COLOR_BGR2RGB))
plt.title('Top Right')
plt.subplot(143)
plt.axis('off')
plt.imshow(cv.cvtColor(bottomLeft, cv.COLOR_BGR2RGB))
plt.title('Bottom Left')
plt.subplot(144)
plt.axis('off')
plt.imshow(cv.cvtColor(bottomRight, cv.COLOR_BGR2RGB))
plt.title('Bottom Right')
plt.tight_layout()
plt.pause(2) 

def process_part(part):
    gray = cv.cvtColor(part.get("slice"), cv.COLOR_BGR2GRAY)

    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    q.put({
        "name": part.get("name"),
        "coords": part.get("coords"),
        "image": gray_bgr
    })

for part in parts:
    t = threading.Thread(target=process_part, args=(part,))
    t.start()
    threads.append(t)

processed = 0

plt.figure(figsize=figsize, dpi=dpi)
while processed < len(parts):
    result = q.get()
    y, x = result.get("coords")
    h, w = result.get("image").shape[:2]

    with lock:
        img[y:y+h, x:x+w] = result["image"]
        print(f"{result.get('name')} received and updated.")
        
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title("Live Update")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(2)  
    
    processed += 1
plt.pause(1)
for t in threads:
    t.join()

plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Final Image")
plt.axis('off')
plt.waitforbuttonpress()