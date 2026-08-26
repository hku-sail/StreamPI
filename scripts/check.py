from torchcodec.decoders import VideoDecoder
import os
import glob


root = "/workspace/datasets/pi/real_world_data_cb408/guess_ball_0422/videos/chunk-000"

splits = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]

for split in splits:
    video_paths = glob.glob(os.path.join(root, split, "*.mp4"))
    for path in video_paths:
        if not os.path.exists(path):
            print(f"不存在: {path}")
            continue
        try:
            decoder = VideoDecoder(path)
            print(f"正常: {path}")
        except Exception as e:
            print(f"损坏: {path} → {e}")