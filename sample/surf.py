import cv2 as cv
import numpy as np
import line_helper
import mat_helper

W = 1164
H = 874
f = 910

intrinsic_mat = np.array([[f, 0, W/2],[0, f, H/2],[0, 0, 1]])

car_mask = 255 * np.concatenate((np.zeros((630, W), np.uint8), np.ones((244, W), np.uint8)))

def main():
  cap = cv.VideoCapture("./labeled/2.hevc")

  surf = cv.xfeatures2d.SURF_create(20)
  bf = cv.BFMatcher()

  ret, old_frame = cap.read()
  old_gray = cv.add(cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY), car_mask)
  old_kps, old_des = surf.detectAndCompute(old_gray, None)

  while cap.isOpened():
    ret, new_frame = cap.read()
    new_gray = cv.add(cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY), car_mask)
    new_kps, new_des = surf.detectAndCompute(new_gray, None)

    matches = bf.knnMatch(old_des, new_des, 2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    matches = good

    # test: points must not go towards the center
    good = []
    center = np.array([W/2, H/2])
    for i, m in enumerate(matches):
      if np.linalg.norm(old_kps[m.queryIdx].pt - center) <= np.linalg.norm(new_kps[m.trainIdx].pt - center):
        good.append(m)
    matches = good

    # test: delete point too far away to be the same point in two consecutive frames
    good = []
    for m in matches:
      if np.linalg.norm(np.subtract(old_kps[m.queryIdx].pt, new_kps[m.trainIdx].pt)) < 20:
        good.append(m)
    matches = good

    old_kps_coord = np.empty((len(matches),2))
    new_kps_coord = np.empty((len(matches),2))

    for i, m in enumerate(matches):
      old_kps_coord[i] = old_kps[m.queryIdx].pt
      new_kps_coord[i] = new_kps[m.trainIdx].pt
    
    # Pose estimation
    E, essential_mask = cv.findEssentialMat(old_kps_coord, new_kps_coord, intrinsic_mat)
    essential_mask = essential_mask.reshape(-1,)

    old_kps_coord = old_kps_coord[essential_mask == 1]
    new_kps_coord = new_kps_coord[essential_mask == 1]

    ret, R, t, pose_mask, world_kps = \
      cv.recoverPose(E, old_kps_coord, new_kps_coord, intrinsic_mat, 100, triangulatedPoints=None)
    pose_mask = pose_mask.reshape(-1,)

    old_kps_coord = old_kps_coord[pose_mask == 255]
    new_kps_coord = new_kps_coord[pose_mask == 255]

    # Angle estimation
    # TODO: correct how it calculates these angles (sometimes it got confused)
    tetay_from_R = mat_helper.get_tetay_from_mat(R)
    tetay_from_t = np.arctan2(t[0], t[2])[0] + np.pi
    tetax_from_t = np.pi - np.arctan2(t[1], t[2])[0]
    print("{:e}".format(tetay_from_t))
    print("{:e}".format(tetax_from_t))
    print("\n")

    # cv.drawMatchesKnn expects list of lists as matches.
    draw_matches = []
    for m in matches:
      draw_matches.append([m])
    draw_frame = cv.drawMatchesKnn(old_frame, old_kps, new_frame, new_kps, draw_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("surf", draw_frame)
    if cv.waitKey(20) & 0xff == ord("q"):
      break


    old_frame = new_frame.copy()
    old_gray = new_gray.copy()
    old_kps = new_kps
    old_des = new_des


  cap.release()
  cv.destroyAllWindows()



if __name__ == "__main__":
  main()
