import argparse
import os

import cv2
import mmcv
import moviepy.editor as mpy
import numpy as np
from torch.nn.modules.utils import _pair


class MMCompact:
    def __init__(self, padding=0.25, threshold=10, hw_ratio=1, allow_imgpad=True):
        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _get_box(self, keypoint, img_shape):
        h, w = img_shape

        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        return (min_x, min_y, max_x, max_y)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f'Could not read first frame from video: {video_path}')
        height, width = frame.shape[:2]

    cap.release()
    return (height, width), fps


def load_annotations(ann_file):
    data = mmcv.load(str(ann_file))
    if isinstance(data, dict) and 'annotations' in data:
        return data['annotations']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported annotation format in {ann_file}: {type(data)}')


def find_annotation(annotations, video_path):

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    for a in annotations:
        if a['frame_dir'] == video_id:
           return a
    raise ValueError(f"Video {video_id} not found in annotation pickle file.")


def draw_markers(frame, box):
    min_x, min_y, max_x, max_y = box
    height, width = frame.shape[:2]

    def clip_point(x, y):
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        return x, y

    tl = clip_point(min_x, min_y)
    tr = clip_point(max_x, min_y)
    bl = clip_point(min_x, max_y)
    br = clip_point(max_x, max_y)

    cv2.circle(frame, tl, 8, (0, 0, 255), -1)
    cv2.circle(frame, tr, 8, (0, 255, 0), -1)
    cv2.circle(frame, bl, 8, (255, 0, 0), -1)
    cv2.circle(frame, br, 8, (0, 255, 255), -1)

    cv2.rectangle(frame, tl, br, (255, 0, 255), 2)
    cv2.putText(frame, 'min_x/min_y', (tl[0] + 10, tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'max_x/min_y', (tr[0] - 180, tr[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, 'min_x/max_y', (bl[0] + 10, bl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, 'max_x/max_y', (br[0] - 180, br[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return frame


def save_video(frames, video_path, output_path, fps):
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'visualize_MMCompose_bbox_{video_id}.mp4')


    if not frames:
        raise ValueError(f'No frames to write for {output_path}')

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    clip = mpy.ImageSequenceClip(frames_rgb, fps=fps)
    clip.write_videofile(str(output_file))
    #clip.write_videofile(str(output_file), codec='libx264', audio=False)
    print(f"Saved visualized video to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Draw the MMCompact corner points on a video.')
    parser.add_argument('--video', required=True, help='Path to the input video')
    parser.add_argument('--ann-file', required=True, help='Annotation pickle that contains the keypoints')
    parser.add_argument('--out', required=True, help='Output video path')
    parser.add_argument('--padding', type=float, default=0.25, help='MMCompact padding value')
    parser.add_argument('--threshold', type=float, default=10, help='MMCompact minimum box size threshold')
    parser.add_argument('--hw-ratio', type=float, default=1.0, help='MMCompact target height/width ratio')
    parser.add_argument('--no-imgpad', dest='allow_imgpad', action='store_false', help='Do not allow the box to extend outside the image')
    parser.set_defaults(allow_imgpad=True)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    ann_file = args.ann_file
    out_path =args.out

    annotations = load_annotations(ann_file)
    annotation = find_annotation(annotations, video_path)

    img_shape, fps = get_video_info(video_path)
    keypoint = np.asarray(annotation['keypoint'], dtype=np.float32)

    compact = MMCompact(
        padding=args.padding,
        threshold=args.threshold,
        hw_ratio=args.hw_ratio,
        allow_imgpad=args.allow_imgpad,
    )
    box = compact._get_box(keypoint, img_shape)
    print(f'Box from keypoints: min_x={box[0]}, min_y={box[1]}, max_x={box[2]}, max_y={box[3]}')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw_markers(frame, box)
        frames.append(frame)

    cap.release()
    save_video(frames, video_path, out_path, fps)
    print(f'Saved marked video to {out_path}')


if __name__ == '__main__':
    main()