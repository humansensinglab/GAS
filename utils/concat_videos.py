import os
import argparse

if __name__ == '__main__':
    # ++++ concat two videos vertically
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_folder1", type=str, default="")
    parser.add_argument("--vid_folder2", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()


    vid_folder1 = args.vid_folder1
    vid_folder2 = args.vid_folder2
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)
    for _, vid_name in enumerate(sorted(os.listdir(vid_folder1))):
        subject_name = vid_name.split("_")[1]
        vid2_name = "subject_" + subject_name + "_grid.mp4"
        vid1 = os.path.join(vid_folder1, vid_name)
        vid2 = os.path.join(vid_folder2, vid2_name)

        save_vid_path = os.path.join(save_path, vid_name)

        os.system(f'ffmpeg -i {vid1} -i {vid2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" -c:v libx264 {save_vid_path}')