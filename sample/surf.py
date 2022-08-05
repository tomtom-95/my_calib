import cv2 as cv
import numpy as np
import line_helper
import mat_helper

W = 1164
H = 874
f = 910

intrinsic_mat = np.array([[f, 0, W/2],[0, f, H/2],[0, 0, 1]])

car_mask = 255 * np.concatenate((np.zeros((600, W), np.uint8), np.ones((274, W), np.uint8)))

center = np.array([W/2, H/2])

class matcher:
  def __init__(self):
    self.surf = cv.xfeatures2d.SURF_create(50)
    self.bf = cv.BFMatcher()

  def kps_des_detect(self, frame):
    gray = cv.add(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), car_mask)
    kps, des = self.surf.detectAndCompute(gray, None)
    return kps, des

  def frame_match(self, old_kps, old_des, new_kps, new_des):
    matches = self.bf.knnMatch(old_des, new_des, 2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches = good

    # test: points must not go towards the center
    good = []
    for m in matches:
      if np.linalg.norm(old_kps[m.queryIdx].pt - center) < np.linalg.norm(new_kps[m.trainIdx].pt - center):
        good.append(m)
    matches = good

    # test: delete points too far away to be the same point in two consecutive frames
    good = []
    for m in matches:
      if np.linalg.norm(np.subtract(old_kps[m.queryIdx].pt, new_kps[m.trainIdx].pt)) < 20:
        good.append(m)
    matches = good

    # test: delete points too close to the corner of the image (way to diminish distorsion effect)
    good = []
    for m in matches:
      if np.linalg.norm(np.subtract(old_kps[m.queryIdx].pt, center)) < 440:
        good.append(m)
    matches = good

    return matches

class drawer:
  def __init__(self):
    return

  def draw(self, matches, frame1, kps1, frame2, kps2):
    # draw
    draw_matches = []
    for m in matches:
      draw_matches.append([m])
    draw_frame = cv.drawMatchesKnn(frame1, kps1, frame2, kps2, draw_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    draw_frame = cv.circle(draw_frame, center.astype(int), 5, (0,255,0), -1)
    return draw_frame


def main():
  cap = cv.VideoCapture("./labeled/0.hevc")

  Matcher = matcher()
  Drawer = drawer()

  frame_cnt = 0
  while cap.isOpened():
    frame_cnt += 1
    if frame_cnt == 1:
      ret, start_frame = cap.read()
      start_kps, start_des = Matcher.kps_des_detect(start_frame)

      old_frame = start_frame.copy()
      old_kps, old_des = start_kps, start_des

    ret, new_frame = cap.read()
    new_kps, new_des = Matcher.kps_des_detect(new_frame)

    if frame_cnt == 1:
      start_matches = Matcher.frame_match(start_kps, start_des, new_kps, new_des)
      matches = start_matches
    else:
      matches = Matcher.frame_match(old_kps, old_des, new_kps, new_des)

    # I want to modify start_matches in such a way that it always has links between the
    # start keypoints and the new keypoints
    good_start_matches = []
    for i, s in enumerate(start_matches):
      for m in matches:
        if s.trainIdx == m.queryIdx:
          dm = cv.DMatch(s.queryIdx, m.trainIdx, 0.01)
          good_start_matches.append(dm)
          break
    start_matches = good_start_matches

    # if start_matches is less than a threshold I must restart the game
    if len(start_matches) < 100:
      start_frame = old_frame.copy()
      start_kps, start_des = Matcher.kps_des_detect(start_frame)
      start_matches = Matcher.frame_match(start_kps, start_des, new_kps, new_des)

    # obtain coords of the matches
    start_kps_coord = np.empty((len(start_matches),2))
    new_kps_coord = np.empty((len(start_matches),2))
    for i, m in enumerate(start_matches):
      start_kps_coord[i] = start_kps[m.queryIdx].pt
      new_kps_coord[i] = new_kps[m.trainIdx].pt
    
    # Pose estimation
    E, essential_mask = cv.findEssentialMat(start_kps_coord, new_kps_coord, intrinsic_mat)
    essential_mask = essential_mask.reshape(-1,)

    start_kps_coord = start_kps_coord[essential_mask == 1]
    new_kps_coord = new_kps_coord[essential_mask == 1]

    ret, R, t, pose_mask, world_kps = \
      cv.recoverPose(E, start_kps_coord, new_kps_coord, intrinsic_mat, 50, triangulatedPoints=None)
    pose_mask = pose_mask.reshape(-1,)

    start_kps_coord = start_kps_coord[pose_mask == 255]
    new_kps_coord = new_kps_coord[pose_mask == 255]

    print(t)

    Drawer.draw(start_matches, start_frame, start_kps, new_frame, new_kps)
    cv.imshow("surf", draw_frame)
    if cv.waitKey(0) & 0xff == ord("q"):
      break

    # update
    old_frame = new_frame.copy()
    old_kps = new_kps
    old_des = new_des


  cap.release()
  cv.destroyAllWindows()


if __name__ == "__main__":
  main()
