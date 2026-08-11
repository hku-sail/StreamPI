#!/usr/bin/env python3

import time

from cv_bridge import CvBridge
import numpy as np
import pyrealsense2 as rs         # Intel RealSense cross-platform open-source API
import rospy

from aloha.msg import RGBGrayscaleImage


TIMEOUT_MS = 100
FPS = 30 #30

cv_bridge = CvBridge()
rospy.init_node('realsense_publisher')

camera_names = ['cam_left_wrist', 'cam_high', 'cam_low', 'cam_right_wrist']
camera_sns = ['130322270931', '128422272318', '218722271368', '130322271696'] #'218622270083', '127122270166']
cam_dict = dict(zip(camera_sns,camera_names))
mean_intensity_set_point_config = { # NOTE these numbers are specific to your lighting setup
    'cam_left_wrist': 500,
    'cam_high': 500,
    'cam_right_wrist': 500,
    'cam_low': 1000
}

print("\n\nSTARTING REALSENSE PUBLISHER\n\n")

ctx = rs.context()
devices = ctx.query_devices()
print([device for device in devices])
device_ids = [d.get_info(rs.camera_info.serial_number) for d in devices]

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()
    color_profiles = []
    depth_profiles = []

    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formats:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        video_type, w, h, fps, fmt))
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

missing_cams = [camera_names[i] for i, c in enumerate(camera_sns) if c not in device_ids]
if missing_cams:
    raise Exception(f"Cameras missing:{missing_cams}")

for dev in devices:
    dev.hardware_reset()
    time.sleep(2)
    # https://dev.intelrealsense.com/docs/high-dynamic-range-with-stereoscopic-depth-cameras#section-2-4-manual-vs-auto-exposure
    # advnc_mode = rs.rs400_advanced_mode(dev)
    # cam_name = cam_dict[dev.get_info(rs.camera_info.serial_number)]
    # intensity_set_point = mean_intensity_set_point_config[cam_name]
    # while True:
    #     # Read-modify-write of the AE control table
    #     ae_ctrl = advnc_mode.get_ae_control()
    #     if ae_ctrl.meanIntensitySetPoint == intensity_set_point:
    #         print('setting meanIntensitySetPoint SUCCESS\n\n')
    #         break
    #     else:
    #         ae_ctrl.meanIntensitySetPoint = intensity_set_point
    #         advnc_mode.set_ae_control(ae_ctrl)
    #         print('attempted setting meanIntensitySetPoint')
    #         time.sleep(0.5)

print(camera_names, camera_sns)

pipes = [rs.pipeline() for _ in range(len(camera_sns))]
cfgs = [rs.config() for _ in range(len(camera_sns))]
profiles = []
depth_scales = []

for cam_name, sn, pipe, cfg in zip(camera_names, camera_sns, pipes, cfgs):
    try:
        print(cam_name, sn)

        cfg.enable_device(sn)
        print(f"Enabled device at {FPS}")
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
        # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
        print("No depth stream")

        # get_profiles()
        print(f"Starting {cam_name}, {sn}")
        profile = pipe.start(cfg)
        device = profile.get_device()

        # depth_sensor = device.first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        profiles.append(profile)
        # depth_scales.append(depth_scale)

    except Exception as e:
        print(f"Error starting {cam_name}, {sn}: {e}")

print('\nWAITING FOR FRAMES\n')
for _ in range(3):
    for pipe, cam_name in zip(pipes, camera_names):
        t = time.time()
        try:
            pipe.wait_for_frames()
            print(f"{cam_name} waited {time.time() - t}s")
        except:
            print(f"{cam_name} waited too long: {time.time() - t}s\n\n")
            raise Exception
        
publishers = [rospy.Publisher(cam_name, RGBGrayscaleImage, queue_size=1) for cam_name in camera_names]

# i = 0

print("\n\nREALSENSE PUBLISHER RUNNING\n\n")
no_error = True
t = time.time()
while not rospy.is_shutdown():
    rgb_imgs = []
    # depth_imgs = []
    msgs = []

    for cam_name, pipe in zip(camera_names, pipes):
        try:
            frameset = pipe.wait_for_frames(timeout_ms=TIMEOUT_MS)
        except:
            print("\n\n", cam_name, "failed\n")
            no_error = False
            [pipe.stop() for pipe in pipes]
            break

        color_frame = np.array(frameset.get_color_frame().get_data())
        # Conver to RGB
        color_frame = color_frame[..., ::-1]
        # depth_frame = np.array(frameset.get_depth_frame().get_data())
        
        rgb_imgs.append(color_frame)
        # depth_imgs.append(depth_frame)

        msg = RGBGrayscaleImage()
        msg.header.stamp = rospy.Time.now()
        msg.images.append(cv_bridge.cv2_to_imgmsg(color_frame, encoding="bgr8"))
        # msg.images.append(cv_bridge.cv2_to_imgmsg(depth_frame, encoding="mono16"))
        msgs.append(msg)

        # h, w = depth_frame.shape
        half_block_size = 5
    
    if not no_error:
        break
        
    for pub, msg in zip(publishers, msgs):
        pub.publish(msg)