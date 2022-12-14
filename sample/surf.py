import cv2 as cv
import numpy as np
import line_helper
import mat_helper
from PySBA import PySBA

W = 1164
H = 874
f = 910

intrinsic_mat = np.array([[f, 0, W/2],[0, f, H/2],[0, 0, 1]])

car_mask = 255 * np.concatenate((np.zeros((600, W), np.uint8), np.ones((274, W), np.uint8)))

center = np.array([W/2, H/2])

class Matcher:
  def __init__(self):
    self.surf = cv.xfeatures2d.SURF_create(10)
    self.bf = cv.BFMatcher()


  def detectAndCompute(self, frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.add(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), car_mask)
    kps, des = self.surf.detectAndCompute(gray, None)

    return kps, des


  def match(self, old_kps, old_des, new_kps, new_des):
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
      if np.linalg.norm(old_kps[m.queryIdx].pt - center) > np.linalg.norm(new_kps[m.trainIdx].pt - center):
        good.append(m)
    matches = good

    # test: delete points too far away to be the same point in two consecutive frames
    good = []
    for m in matches:
      if np.linalg.norm(np.subtract(old_kps[m.queryIdx].pt, new_kps[m.trainIdx].pt)) < 25:
        good.append(m)
    matches = good

    # test: delete points too close to the corner of the image (way to diminish distorsion effect)
    good = []
    for m in matches:
      if np.linalg.norm(np.subtract(old_kps[m.queryIdx].pt, center)) < 450:
        good.append(m)
    matches = good

    return matches


class Drawer:
  def __init__(self):
    return


  def draw(self, matches, frame1, kps1, frame2, kps2):
    draw_matches = []
    for m in matches:
      draw_matches.append([m])
    draw_frame = cv.drawMatchesKnn(frame1, kps1, frame2, kps2, draw_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    draw_frame = cv.circle(draw_frame, center.astype(int), 5, (0,255,0), -1)
    return draw_frame


class Poser:
  def __init__(self):
    return


  def pose(self, matches, kps1, kps2):
    coord1 = np.empty((len(matches),2))
    coord2 = np.empty((len(matches),2))
    for i, m in enumerate(matches):
      coord1[i] = kps1[m.queryIdx].pt
      coord2[i] = kps2[m.trainIdx].pt

    # Pose estimation
    E, essential_mask = cv.findEssentialMat(coord1, coord2, intrinsic_mat)
    essential_mask = essential_mask.reshape(-1,)

    coord1 = coord1[essential_mask == 1]
    coord2 = coord2[essential_mask == 1]

    ret, R, t, pose_mask, world_kps = \
      cv.recoverPose(E, coord1, coord2, intrinsic_mat, 100, triangulatedPoints=None)
    pose_mask = pose_mask.reshape(-1,)

    coord1 = coord1[pose_mask == 255]
    coord2 = coord2[pose_mask == 255]

    # from funky notation to numpy array for 3-dim points in homogeneous coords
    good_world_kps = np.empty((len(world_kps[0]), 4))
    for i, dim in enumerate(world_kps):
      for j, val in enumerate(world_kps[i]):
        good_world_kps[j,i] = val
    world_kps = good_world_kps

    return R, t, world_kps


def main():
  cap = cv.VideoCapture("./labeled/0_rev.mp4")

  matcher = Matcher()
  poser = Poser()
  drawer = Drawer()
  #sba = PySBA()

  ret, start_frame = cap.read()
  start_kps, start_des = matcher.detectAndCompute(start_frame)

  old_frame = start_frame.copy()
  old_kps, old_des = start_kps, start_des

  rotations = []
  translations = []
  camera_cnt = 0
  while cap.isOpened():
    ret, new_frame = cap.read()
    new_kps, new_des = matcher.detectAndCompute(new_frame)

    old_new_matches = matcher.match(old_kps, old_des, new_kps, new_des)

    try: 
      start_matches
    except NameError:
      start_matches = matcher.match(start_kps, start_des, new_kps, new_des)
      old_new_matches = start_matches

    good_start_matches = []
    for s in start_matches:
      for m in old_new_matches:
        if s.trainIdx == m.queryIdx:
          good_start_matches.append(cv.DMatch(s.queryIdx, m.trainIdx, 0.001))
          break
    start_matches = good_start_matches

    # if start_matches is less than a threshold I must restart the game
    if len(start_matches) < 100:
      R, t, world_kps = poser.pose(start_matches, start_kps, new_kps)

      eucl_world_kps = cv.convertPointsFromHomogeneous(world_kps)
      """
      use the 3dim keypoint as input in PYSBA. This points are the initial estimate of the positions. 
      """
      """
      to use bundle adjustment I must store information about the rotation and translation frame by frame
      not just use the information when the start matches are too low
      """
      sba = PySBA()

      start_frame = old_frame.copy()
      start_kps, start_des = matcher.detectAndCompute(start_frame)
      start_matches = matcher.match(start_kps, start_des, new_kps, new_des)

    draw_frame = drawer.draw(start_matches, start_frame, start_kps, new_frame, new_kps)
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
