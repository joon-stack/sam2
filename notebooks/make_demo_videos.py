import cv2
import numpy as np
import os


def save_video_and_frames(data, video_path, frames_folder, fps=30):
    """
    Save a video from a list of images and also save individual frames as JPG files.

    Args:
        data (list): List containing dictionaries with image data under 'image' key.
        video_path (str): Path to save the resulting video.
        frames_folder (str): Directory to save the individual frames.
        fps (int): Frames per second for the video.
    """
    # Ensure the frames folder exists
    os.makedirs(frames_folder, exist_ok=True)

    # Extract the first image to get the shape
    first_image = data[0]['image']
    height, width, channels = first_image.shape  # Assume all images have the same shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each frame to the video and save as JPG
    for idx, frame in enumerate(data):
        image = frame['image']  # Extract the image

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image.max() > 1:  # Ensure the image is in uint8 format
            image = image.astype(np.uint8)

        # Save the frame to the video
        video_writer.write(image)


        # Save the frame as a JPG file
        frame_path = os.path.join(frames_folder, f"{idx:05d}.jpg")
        cv2.imwrite(frame_path, image)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {video_path}")
    print(f"Frames saved to {frames_folder}")

# Example usage
video_path = "sam2/notebooks/videos/pointmaze.mp4"
x = np.load('/home/s2/youngjoonjeong/github/dino-wm/dataset/pointmaze/point_maze_val.npy', allow_pickle=True)
frames_folder = "/home/s2/youngjoonjeong/github/sam2/notebooks/videos/pointmaze"
save_video_and_frames(x[0], video_path, frames_folder, fps=30)