import time
import cv2
from eyeconnect.gaze.capture import Camera
from eyeconnect.gaze.facemesh import FaceTracker

cam = Camera()
tr = FaceTracker()
print("backend=", tr.backend, flush=True)
N = 150
det = 0
best = None
bestconf = -1
bfeat = None
bdbg = {}
last = None
lastdbg = {}
t0 = time.time()
for i in range(N):
    f = cam.read()
    if f is None:
        continue
    last = f
    feat, conf, dbg = tr.process(f)
    lastdbg = dbg
    if feat is not None:
        det += 1
        if conf > bestconf:
            bestconf = conf
            best = f.copy()
            bfeat = feat
            bdbg = dbg
print(f"frames={N} det={det} elapsed={time.time()-t0:.1f}s fps={cam.fps:.1f}", flush=True)
out = best if best is not None else last
if out is not None:
    dbg = bdbg if best is not None else lastdbg
    for k in ("l_iris", "r_iris"):
        if k in dbg:
            x, y = dbg[k]
            cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
    if best is not None:
        msg = f"det {det}/{N} feat={bfeat[0]:.2f},{bfeat[1]:.2f}"
    else:
        msg = f"det 0/{N} NO FACE - syad pered kameroy v 60sm"
    cv2.putText(out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(r"C:\Temp\opencode\webcam_check.jpg", out)
    print("saved C:/Temp/opencode/webcam_check.jpg", flush=True)
cam.release()
tr.close()
