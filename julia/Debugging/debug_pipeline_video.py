import argparse
import copy
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.utils import build_from_cfg

import pyskl.datasets.pipelines  # noqa: F401
from pyskl.datasets.builder import PIPELINES


def get_video_info(video_path):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise ValueError(f'Could not open video: {video_path}')

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	fps = fps if fps and fps > 0 else 24
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	if frame_count <= 0 or width <= 0 or height <= 0:
		frame_count = 0
		ok, frame = cap.read()
		if not ok:
			raise ValueError(f'Could not read first frame from video: {video_path}')
		height, width = frame.shape[:2]
		frame_count = 1
		while True:
			ok, _ = cap.read()
			if not ok:
				break
			frame_count += 1

	cap.release()
	return frame_count, (height, width), fps


def load_annotations(ann_file):
	data = mmcv.load(str(ann_file))
	if isinstance(data, dict) and 'annotations' in data:
		return data['annotations']
	if isinstance(data, list):
		return data
	raise TypeError(f'Unsupported annotation format in {ann_file}: {type(data)}')


def find_annotation(annotations, video_path):
	video_stem = Path(video_path).stem
	video_name = Path(video_path).name

	for item in annotations:
		for key in ('filename', 'frame_dir'):
			value = item.get(key)
			if value is None:
				continue
			value_path = Path(str(value))
			if value_path.stem == video_stem or value_path.name == video_name:
				return item

	raise ValueError(
		f'No annotation entry found for video {video_path}. '
		'Check the ann_file or the video name.')


def prepare_results(annotation, video_path, frame_count, img_shape):
	results = copy.deepcopy(annotation)
	results['filename'] = str(video_path)
	results['frame_dir'] = Path(video_path).stem
	results['total_frames'] = frame_count
	results['img_shape'] = img_shape
	results['original_shape'] = img_shape
	results['modality'] = 'RGB'
	results['start_index'] = 0
	results['test_mode'] = False
	return results


def build_pipeline(pipeline_cfg):
	steps = []
	for step_cfg in pipeline_cfg:
		steps.append(build_from_cfg(step_cfg, PIPELINES))
	return steps


def denormalize_images(imgs, mean, std):
	imgs = imgs.detach().cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)

	if imgs.ndim == 5:
		return [denormalize_images(imgs[i], mean, std) for i in range(imgs.shape[0])]

	if imgs.ndim != 4:
		raise ValueError(f'Expected imgs with 4 or 5 dimensions, got {imgs.shape}')

	if imgs.shape[0] == 3:
		frames = np.transpose(imgs, (1, 2, 3, 0))
	elif imgs.shape[1] == 3:
		frames = np.transpose(imgs, (0, 2, 3, 1))
	else:
		raise ValueError(f'Could not infer channel dimension from imgs shape {imgs.shape}')

	frames = frames.astype(np.float32)
	mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
	std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
	frames = frames * std + mean
	frames = np.clip(frames, 0, 255).astype(np.uint8)
	return [frame for frame in frames]


def save_video(frames, output_path, fps):
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if len(frames) == 0:
		raise ValueError(f'No frames to write for {output_path}')

	height, width = frames[0].shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
	if not writer.isOpened():
		raise ValueError(f'Could not open VideoWriter for {output_path}')

	for frame in frames:
		writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	writer.release()


def parse_args():
	parser = argparse.ArgumentParser(
		description='Visualize a video after the exact train_pipeline preprocessing.')
	parser.add_argument('--config', required=True, help='Path to the pyskl config file')
	parser.add_argument('--video', required=True, help='Path to the input video')
	parser.add_argument('--ann-file', default=None, help='Annotation file used to find the matching sample')
	parser.add_argument('--out', required=True, help='Output video path or output directory if multiple clips are produced')
	parser.add_argument('--show-steps', action='store_true', help='Print the instantiated pipeline modules')
	parser.add_argument('--trace', action='store_true', help='Print shapes after each preprocessing step')
	return parser.parse_args()


def main():
	args = parse_args()
	cfg = Config.fromfile(args.config)

	train_pipeline = getattr(cfg, 'train_pipeline', None)
	if train_pipeline is None:
		raise AttributeError('Config does not define a top-level train_pipeline')

	ann_file = args.ann_file or getattr(cfg, 'ann_file', None)
	if ann_file is None:
		raise AttributeError('No annotation file provided and config does not define ann_file')

	annotations = load_annotations(ann_file)
	annotation = find_annotation(annotations, args.video)

	frame_count, img_shape, fps = get_video_info(args.video)
	if 'total_frames' in annotation and annotation['total_frames'] != frame_count:
		print(f"Warning: annotation total_frames={annotation['total_frames']} but video has {frame_count} frames.")

	pipeline = build_pipeline(train_pipeline)
	if args.show_steps:
		print('Pipeline modules:')
		for idx, step in enumerate(pipeline):
			print(f'  {idx:02d}: {step}')

	results = prepare_results(annotation, args.video, frame_count, img_shape)

	if args.trace:
		print(f"initial: keys={sorted(results.keys())}, img_shape={results['img_shape']}, total_frames={results['total_frames']}")

	for idx, step in enumerate(pipeline):
		results = step(results)
		if results is None:
			raise RuntimeError(f'Pipeline returned None at step {idx}: {step}')
		if args.trace:
			img_shape_info = results.get('img_shape', None)
			if 'imgs' in results:
				imgs_info = getattr(results['imgs'], 'shape', None)
			else:
				imgs_info = None
			print(f'after {idx:02d} {step.__class__.__name__}: img_shape={img_shape_info}, imgs={imgs_info}, keys={sorted(results.keys())}')

	if 'imgs' not in results:
		raise KeyError('Pipeline output does not contain imgs')

	if not hasattr(cfg, 'img_norm_cfg'):
		raise AttributeError('Config does not define img_norm_cfg')

	frames = denormalize_images(results['imgs'], cfg.img_norm_cfg['mean'], cfg.img_norm_cfg['std'])

	out_path = Path(args.out)
	if isinstance(frames, list) and frames and isinstance(frames[0], list):
		if out_path.suffix:
			out_dir = out_path.parent
			stem = out_path.stem
			suffix = out_path.suffix
		else:
			out_dir = out_path
			stem = Path(args.video).stem
			suffix = '.mp4'

		out_dir.mkdir(parents=True, exist_ok=True)
		for clip_idx, clip_frames in enumerate(frames):
			clip_path = out_dir / f'{stem}_clip{clip_idx}{suffix}'
			save_video(clip_frames, clip_path, fps)
		print(f'Saved {len(frames)} clips to {out_dir}')
		return

	if len(frames) == 0:
		raise ValueError('No frames produced by the pipeline')

	save_video(frames, out_path, fps)
	print(f'Saved pipeline output to {out_path}')


if __name__ == '__main__':
	main()


#python julia/Debugging/debug_pipeline_video.py --config julia/Debugging/rgb_only_debugging.py --video ../WLASL300/WLASL_300_compressed/test/language/32167.mp4 --ann-file julia/WLASL300/pyskl_mediapipe_annos_2d_denormalized_NO_KPS_FROM_BODYMODEL.pkl --out ../pipelineVisualization