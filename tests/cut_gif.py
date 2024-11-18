from PIL import Image
from tqdm import tqdm
# import imageio


def split_gif(gif_path):
    frames = []
    with Image.open(gif_path) as img:
        img.seek(0)
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames


def cut_image(image, coordinates):
    return image.crop(coordinates)


def save_image(image, path):
    image.save(path)


# Example usage
gif_frames = split_gif(
    'results_gps/23_07_31_12_15/sample_rollouts/animation_2.gif')
i = 740
# Assuming you want to cut the 5th frame
# for i in tqdm([2, 124, 374, 749], desc='cutting frames', unit='frame'):
frame_to_cut = gif_frames[i]  # Index is zero-based
# Define the coordinates to cut
coordinates_to_cut = (325, 75, 1250, 740)  # Define your coordinates
cut_image = cut_image(frame_to_cut, coordinates_to_cut)
# Save the cut image
save_image(
    cut_image, f'results_gps/23_07_31_12_15/sample_rollouts/animation_2_{i}.png')
