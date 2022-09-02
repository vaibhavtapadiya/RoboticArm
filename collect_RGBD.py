import pybullet as p
import time
from random import *
import numpy as np
import pybullet_data
import os
import numpngw

# environment urdf path
plane_path = 'data/plane/plane.urdf'
robot_path = 'data/sawyer_robot/sawyer_description/urdf/sawyer.urdf'

clid = p.connect(p.GUI)
if (clid < 0):
    p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load plane and the robot
planeId = p.loadURDF(plane_path, [0, 0, -1])
sawyerId = p.loadURDF(robot_path, [0, 0, 0],
                      [0, 0, 0, 3])


def get_RGBD_single_object(object_path, init_position, init_orientation, save_path):
    """
    get rgb and depth images for a single object
    Args:
        init_x_angle: the initial angle of along x-axis
        init_y_angle: the initial angle of along y-axis
        init_z_angle: the initial angle of along z-axis
        object_path: path for the objects
        init_x: the initial position of x
        init_y: the initial position of y
        init_z: the initial position of z
        save_path: the save directory for rgb and depth

    Returns: save rgb and depth images for a single object

    """

    count = 0
    while count < 50:
        print("count",count)
        # load object in pybullet environment
        init_pos_xyz = init_position  # initial position x,y,z
        init_ori_xyz = init_orientation  #orientation x,y,z

        pos_xyz = [init_pos_xyz[0] + 0.002 * count,
                   init_pos_xyz[1] + 0.002 * count,
                   init_pos_xyz[2] + 0.002 * count]

        ori_xyz = [init_ori_xyz[0] + 0.2 * count,
                   init_ori_xyz[1] + 0.2 * count,
                   init_ori_xyz[2] + 0.2 * count]

        objectID = p.loadURDF(object_path,pos_xyz,p.getQuaternionFromEuler(ori_xyz))
        #
        # # apply texture for the object
        # texture_id = p.loadTexture(texture_path)
        # p.changeVisualShape(objectID,-1,textureUniqueId=texture_id)

        # Using the inserted camera to caputure data for training. Save the captured numpy array as image files for later training process.

        width = 256
        height = 256

        fov = 60
        aspect = width / height
        near = 0.02
        far = 1

        # the view_matrix should contain three arguments, the first one is the [X,Y,Z] for camera location
        #												  the second one is the [X,Y,Z] for target location
        #											      the third one is the  [X,Y,Z] for the up-vector of the camera
        # Example:
        # viewMatrix = pb.computeViewMatrix(
        #     cameraEyePosition=[0, 0, 3],
        #     cameraTargetPosition=[0, 0, 0],
        #     cameraUpVector=[0, 1, 0])

        # view_matrix = p.computeViewMatrix([1.05, -0.05, 0.4],
        #                                   pos_xyz,
        #                                   [0, 0, 0])

        view_matrix = p.computeViewMatrix([1.05,-0.05,0.4],
                                          init_pos_xyz,
                                          [20,0,1])

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # Get depth values using the OpenGL renderer
        images = p.getCameraImage(width,
                                  height,
                                  view_matrix,
                                  projection_matrix,
                                  shadow=True,
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

        # time.sleep(1)

        # save depth_image
        depth_image = depth_opengl
        z_depth_image = (65535*((depth_image - depth_image.min())/depth_image.ptp())).astype(np.uint16)
        depth_temp_path = save_path + '/depth/'
        if os.path.exists(os.path.join(os.getcwd(),depth_temp_path)):
            numpngw.write_png(depth_temp_path+str(count)+'.png',z_depth_image)
        else:
            os.makedirs(os.path.join(os.getcwd(),depth_temp_path))
            numpngw.write_png(depth_temp_path+str(count) + '.png', z_depth_image)
        p.resetBasePositionAndOrientation(objectID, [2, 2, 2], [0, 0, 0, 1])
        count+=1
        print("after_count", count)

    return


if __name__ == "__main__":
    # object paths
    object_path = []
    object_path.append('data/random_urdfs/040/040.urdf')
    # object_path.append('data/Pablo_object/171/171.urdf')
    # object_path.append('data/Pablo_object/196/196.urdf')

    single_object_save_path = 'training_data_test/'
    double_objects_save_path = 'training_data_test/'
    


    for i in range(len(object_path)):
        get_RGBD_single_object(object_path[i],
                             [1.1,0.05,0],
                             [0,0,1.56],
                             single_object_save_path+str(i))


                
    i = 0
    while 1:
        i += 1
        p.stepSimulation()
        # 0.03 sec/frame
        time.sleep(0.03)
        # increase i to increase the simulation time
        if (i == 20000000):
            break





        



































































































































































































































































































