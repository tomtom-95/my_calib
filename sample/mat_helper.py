"""
Obtain angles of rotations from a rotation matrix
Convention: right-handed reference frame/coordinate system (that's all you need to say to know 
how rotation aroud a certain axes works)
The rotation matrix is a unique description of the change of the reference frame (not sure 
what comma is talking about when they say the rotation matrix is somewhat amibigous)
What is subject to ambiguity is the kind of operation we read from the rotation matrix to get 
the angles of rotations
We must specify exactly how the rotation is executed, what change and what remains fixed and 
in what order the rotation around the axes are executed
I think the rotation matrix as applied to the coordinates of a point, this has brought me a lot 
of confusion!
To be precise: the rotation matrix is applied to the coordinates of a point P to obtain the 
coordinates of the point in the new reference frame. The point remains exactly where it is!
It does not move. What moves is the reference frame. 
In the new reference frame the point P is described by the new coordinates.
This is exactly what happens with the dashcam. The dashcam moves, so the reference frame moves.
The object in the scene remains where they are!
This is clearly described by:
	https://mathworld.wolfram.com/RotationMatrix.html

After stating that we have a clear interpretation of the angles we are going to obtain
The rotations are done around the x,y,z axes of the origin frame of reference in this order:
	rotation around the x axes by an angle teta_x
	rotation around the y axes by an angle teta_y
	rotation around the z axes by an angle teta_z
The rotations are applied wrt the old reference frame
"""
import numpy as np

def get_tetax_from_mat(rot_mat):
	return -np.arctan2(rot_mat[2][1], rot_mat[2][2])

def get_tetay_from_mat(rot_mat):
	return np.arcsin(rot_mat[2][0])

def get_tetaz_from_mat(rot_mat):
	return -np.arctan2(rot_mat[1][0], rot_mat[0][0])
