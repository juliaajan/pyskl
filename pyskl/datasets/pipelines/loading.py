# Copyright (c) OpenMMLab. All rights reserved.
import io
import numpy as np
import os.path as osp
from mmcv.fileio import FileClient

from ..builder import PIPELINES


@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        num_threads (int): Number of thread to decode the video. Default: 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', num_threads=1, label_mapping_file=None, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None
        self.label_mapping = {}

        if label_mapping_file:
            self.label_mapping = self._load_label_mapping(label_mapping_file)

    def _load_label_mapping(self, filepath):
        """Load label_mapping.txt and create a dict with {label_id: label_name}."""
        label_map = {}
        with open(filepath, 'r') as f:
            for line in f:
                idx, label_name = line.strip().split('\t')
                label_map[int(idx)] = label_name
        return label_map

    def _get_videoreader(self, filename):
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=1)
        try:
            return container
        except Exception as e:
            print(f"Error occurred while loading video: {filename}: {e}")
            raise ValueError(f"Error occurred while loading video: {filename}: {e}")

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'filename' not in results:
            assert 'frame_dir' in results
            if self.label_mapping:
                #add split and label to filename, as WLASL300 videos are in nested folders: 'split/label/12345.mp4' 
                split = results['split']
                old_path = results['frame_dir'] #e.g. ../WLASL300/WLASL_300_compressed/10260
                path_prefix, video_id = old_path.rsplit('/', 1)
                label_id = results['label']
                label_name = self.label_mapping.get(label_id)
                results['filename'] = path_prefix + '/' + split + '/' + label_name + '/' + video_id + '.mp4'               
            else:
                results['filename'] = video_id + '.mp4'
        try:
            results['video_reader'] = self._get_videoreader(results['filename'])
        except Exception as e:
            print(f"Error occurred while loading video: {results['filename']}: {e}")
            raise ValueError(f"Error occurred while loading video: {results['filename']}: {e}")

        if 'total_frames' in results:
            #tolerate only 1 frame difference due to compression
            assert abs(results['total_frames'] - len(results['video_reader'])) <= 1, (
                'SkeFrames', results['total_frames'], 'VideoFrames', len(results['video_reader']), 'for video ', results['filename']
            )
        else:
            results['total_frames'] = len(results['video_reader'])        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets. Default: 'accurate'.
    """

    def __init__(self, mode='accurate'):
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container, frame_inds):
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class ArrayDecode:
    """Load and decode frames with given indices from a 4D array.

    Required keys are "array and "frame_inds", added or modified keys are
    "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        modality = results['modality']
        array = results['array']

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for i, frame_idx in enumerate(results['frame_inds']):

            frame_idx += offset
            if modality == 'RGB':
                imgs.append(array[frame_idx])
            elif modality == 'Flow':
                imgs.extend(
                    [array[frame_idx, ..., 0], array[frame_idx, ..., 1]])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
