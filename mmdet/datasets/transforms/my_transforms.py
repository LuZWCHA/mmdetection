# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type
from .transforms import Mosaic

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]


@TRANSFORMS.register_module()
class CachedMosaic2Images(Mosaic):
    """Cached mosaic augmentation.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The cached mosaic transform steps are as follows:

         1. Append the results from the last transform into the cache.
         2. Choose the mosaic center as the intersections of 4 images
         3. Get the left top image according to the index, and randomly
            sample another 3 images from the result cache.
         4. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (np.float32) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results_cache = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, 'The length of cache must >= 4, ' \
                                       f'but got {max_cached_images}.'
        self.max_cached_images = max_cached_images

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if 'gt_masks' in results else False

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
            mosaic_img2 = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img2'].dtype)

        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)
            mosaic_img2 = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img2'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            img_i = results_patch['img']
            img_i2 = results_patch['img2']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])

            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            img_i2 = mmcv.imresize(
                img_i2, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            # compute the combine parameters
            paste_coord2, crop_coord2 = self._mosaic_combine(
                loc, center_position, img_i2.shape[:2][::-1])

            x1_p, y1_p, x2_p, y2_p = paste_coord2
            x1_c, y1_c, x2_c, y2_c = crop_coord2
            mosaic_img2[y1_p:y2_p, x1_p:x2_p] = img_i2[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal')
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2),
                               int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img2'] = mosaic_img2
        results['img_shape'] = mosaic_img.shape[:2]
        results['img_shape2'] = mosaic_img2.shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        if with_mask:
            mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
            results['gt_masks'] = mosaic_masks[inside_inds]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop})'
        return repr_str


@TRANSFORMS.register_module()
class BndboxShift(BaseTransform):
    """Shift the box given shift pixels and probability.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32])

    Modified Keys:

    - img
    - gt_bboxes

    Args:
        prob (float): Probability of shifts. Defaults to 0.5.
        max_shift_px (int): The max pixels for shifting. Defaults to 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Defaults to 1.
        unchange_thr_px (int): The width and height threshold for unchanging.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_shift_px: int = 32,
                 filter_thr_px: int = 1,
                 unchange_thr_px: int = 32) -> None:
        assert 0 <= prob <= 1
        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        self.unchange_thr_px = int(unchange_thr_px)

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to random shift bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        gt_bboxes = results['gt_bboxes']
        img_shape = results['img'].shape[:2]

        for i in range(len(gt_bboxes)):
            gt_bbox = gt_bboxes[i].clone()

            if (gt_bbox.cxcywh[0, 2:]).min() > self.unchange_thr_px and self._random_prob() < self.prob:
                random_shift_x = random.randint(-self.max_shift_px,
                                                self.max_shift_px)
                random_shift_y = random.randint(-self.max_shift_px,
                                                self.max_shift_px)
                gt_bbox.translate_([random_shift_x, random_shift_y])

                # clip border
                gt_bbox.clip_(img_shape)

                if gt_bbox.cxcywh[0, 2:].min() > self.filter_thr_px:
                    gt_bboxes[i] = gt_bbox
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_shift_px={self.max_shift_px}, '
        repr_str += f'filter_thr_px={self.filter_thr_px}, '
        repr_str += f'unchange_thr_px={self.unchange_thr_px})'
        return repr_str

