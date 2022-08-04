"""
This is completely shit and I do not know why
"""
import cv2 as cv
import numpy as np
import line_helper
import mat_helper

W = 1164
H = 874
f = 910

intrinsic_mat = np.array([[f, 0, W/2],[0, f, H/2],[0, 0, 1]])

car_mask = 255 * np.concatenate((np.zeros((645, W), np.uint8), np.ones((229, W), np.uint8)))

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize = (25, 25),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
)

def main():
  cap = cv.VideoCapture("./labeled/0.hevc")

  surf = cv.xfeatures2d.SURF_create(10)

  ret, old_frame = cap.read()
  old_gray = cv.add(cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY), car_mask)
  old_kps, old_des = surf.detectAndCompute(old_gray, None)

  old_coord = np.empty((len(old_kps),2), dtype=np.float32)
  for i, old in enumerate(old_kps):
    old_coord[i] = old.pt

  start_kps, start_des = old_kps, old_des
  start_coord = old_coord
  start_frame = old_frame.copy()
  start_gray = old_gray.copy()

  while cap.isOpened():
    ret, new_frame = cap.read()
    new_gray = cv.add(cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY), car_mask)
    new_kps, new_des = surf.detectAndCompute(new_gray, None)

    new_coord, status, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, old_coord, None, **lk_params)
    status = status.reshape(-1,)

    start_coord = start_coord[status == 1]
    old_coord = old_coord[status == 1]
    new_coord = new_coord[status == 1]

    status = np.ones((len(start_coord),))
    for i, (old, new) in enumerate(zip(old_coord, new_coord)):
      if np.linalg.norm(old - new) > 5:
        status[i] = 0

    start_coord = start_coord[status == 1]
    old_coord = old_coord[status == 1]
    new_coord = new_coord[status == 1]

    status = np.ones((len(start_coord),))
    center = np.array([W/2, H/2])
    for i, (old, new) in enumerate(zip(old_coord, new_coord)):
      if np.linalg.norm(old - center) > np.linalg.norm(new - center):
        status[i] = 0

    start_coord = start_coord[status == 1]
    old_coord = old_coord[status == 1]
    new_coord = new_coord[status == 1]


    # Pose estimation
    E, essential_mask = cv.findEssentialMat(start_coord, new_coord, intrinsic_mat)
    essential_mask = essential_mask.reshape(-1,)

    start_coord = start_coord[essential_mask == 1]
    old_coord = old_coord[essential_mask == 1]
    new_coord = new_coord[essential_mask == 1]

    ret, R, t, pose_mask, world_kps = \
      cv.recoverPose(E, start_coord, new_coord, intrinsic_mat, 200, triangulatedPoints=None)
    pose_mask = pose_mask.reshape(-1,)

    start_coord = start_coord[pose_mask == 255]
    old_coord = old_coord[pose_mask == 255]
    new_coord = new_coord[pose_mask == 255]

    # draw
    draw_mask = np.zeros_like(new_frame)
    draw_frame = new_frame.copy()
    for i, (new, old) in enumerate(zip(new_coord, start_coord)):
        a, b = new.ravel()
        c, d = old.ravel()
        draw_mask = cv.line(draw_mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
        draw_frame = cv.circle(draw_frame, (int(a), int(b)), 5, (0,0,255), -1)
    draw_frame = cv.add(draw_frame, draw_mask)

    if len(start_coord) < 50:
      t = -t
      tetay_from_t = np.arctan2(t[0], t[2])
      print(tetay_from_t)
      start_frame = new_frame.copy()
      start_gray = new_gray.copy()
      start_kps, start_des = surf.detectAndCompute(start_gray, None)

      start_coord = np.empty((len(start_kps),2), dtype=np.float32)
      for i, start in enumerate(start_kps):
        start_coord[i] = start.pt

      new_coord, status, err = cv.calcOpticalFlowPyrLK(old_gray, start_gray, start_coord, None, **lk_params)
      status = status.reshape(-1,)

      start_coord = start_coord[status == 1]
      new_coord = new_coord[status == 1]

    cv.imshow("frame", draw_frame)
    if cv.waitKey(10) & 0xff == ord("q"):
      break

    old_frame = new_frame.copy()
    old_gray = new_gray.copy()
    old_kps = new_kps
    old_des = new_des
    old_coord = new_coord


  cap.release()
  cv.destroyAllWindows()



if __name__ == "__main__":
  main()
