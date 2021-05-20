import argparse
import os
from PIL import Image, ImageSequence

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gif1', type=str, required=True, help='path to first gif')
    parser.add_argument('--gif2', type=str, required=True, help='path to second gif')
    parser.add_argument('--out', type=str, required=True, help='path to output gif')
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.gif1):
        raise AssertionError(f"{args.gif1} not found.")
    if not os.path.isfile(args.gif2):
        raise AssertionError(f"{args.gif2} not found.")

    gif1 = Image.open(args.gif1)
    gif2 = Image.open(args.gif2)
    print(f'gif1 has: {gif1.n_frames} frames')
    print(f'gif2 has: {gif2.n_frames} frames')

    imgs = []
    for frame in range(gif1.n_frames):
        gif1.seek(frame)
        imgs.append(gif1.convert('RGBA'))
    for frame in range(gif2.n_frames):
        gif2.seek(frame)
        imgs.append(gif2.convert("RGBA"))
    
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], 
                optimize=False, duration=100, loop=0)

if __name__ == '__main__':
    main()