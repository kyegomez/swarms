from typing import List

import cv2


def video_to_frames(video_file: str) -> List:
    """
    Convert a video into frames.

    Args:
        video_file (str): The path to the video file.

    Returns:
        List[np.array]: A list of frames from the video.
    """
    # Open the video file
    vidcap = cv2.VideoCapture(video_file)

    frames = []
    success, image = vidcap.read()

    while success:
        frames.append(image)
        success, image = vidcap.read()

    return frames


def save_frames_as_images(frames, output_dir) -> None:
    """
    Save a list of frames as image files.

    Args:
        frames (list of np.array): The list of frames.
        output_dir (str): The directory where the images will be saved.
    """
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_dir}/frame{i}.jpg", frame)


# out = save_frames_as_images(frames, "playground/demos/security_team/frames")

# print(out)
