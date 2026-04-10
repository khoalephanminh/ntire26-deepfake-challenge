#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: algohunt
# Microsoft Research & Peking University 
# lilingzhi@pku.edu.cn
# Copyright (c) 2019

#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import inspect
import logging
import sys

import cv2
import numpy as np

# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = sorted([name for name, obj in inspect.getmembers(sys.modules[__name__])
                    if inspect.isclass(obj) and name != "Mask"])
    masks.append("none")
    # logger.debug(masks)
    return masks


def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "dfl_full"
    default = default if default in masks else masks[0]
    # logger.debug(default)
    return default


class Mask():
    """ Parent class for masks
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, channels=4):
        # logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
        #              self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        #logger.info("Initialized %s", self.__class__.__name__)

    def build_mask(self):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        #logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        #logger.info("Final mask shape: %s", retval.shape)
        return retval


class dfl_full(Mask):  # pylint: disable=invalid-name
    """ DFL facial mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        jaw = (self.landmarks[0:17],
               self.landmarks[48:68],
               self.landmarks[0:1],
               self.landmarks[8:9],
               self.landmarks[16:17])
        eyes = (self.landmarks[17:27],
                self.landmarks[0:1],
                self.landmarks[27:28],
                self.landmarks[16:17],
                self.landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class components(Mask):  # pylint: disable=invalid-name
    """ Component model mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
        l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
        r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
        l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
        nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
        r_eye = (self.landmarks[17:22],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        l_eye = (self.landmarks[22:27],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        nose = (self.landmarks[27:31], self.landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask

class inner_components(Mask):  # pylint: disable=invalid-name
    """ Component model mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        # r_eye = (self.landmarks[17:22],
        #          self.landmarks[27:28],
        #          self.landmarks[31:36])
        # l_eye = (self.landmarks[22:27],
        #          self.landmarks[27:28],
        #          self.landmarks[31:36])
        # nose = (self.landmarks[27:31], self.landmarks[31:36])
        # mouth = (self.landmarks[48:60], self.landmarks[60:68])
        # parts = [r_eye, l_eye, nose, mouth]
        r_eye = (self.landmarks[17:22],
                 self.landmarks[27:28],
                 self.landmarks[31:36])
        l_eye = (self.landmarks[22:27],
                 self.landmarks[27:28],
                 self.landmarks[31:36])
        nose = (self.landmarks[27:31], self.landmarks[31:36])
        mouth = (self.landmarks[48:60], self.landmarks[60:68])
        x_48, y_48 = self.landmarks[48]
        x_54, y_54 = self.landmarks[54]
        x_60, y_60 = self.landmarks[60]
        x_64, y_64 = self.landmarks[64]
        x_17, y_17 = self.landmarks[17]
        x_26, y_26 = self.landmarks[26]
        wider_mouth = (np.array([((x_48 + x_17) / 2, y_48)], dtype=np.int32), np.array([((x_54 + x_26) / 2, y_54)], dtype=np.int32))

        average_y = (self.landmarks[8][1] - self.landmarks[57][1]) / 2
        adjusted_mouth = np.copy(self.landmarks[48:60])
        adjusted_mouth[:, 1] += average_y.astype(adjusted_mouth.dtype)
        mouth = (adjusted_mouth, self.landmarks[60:68])

        higher_eyebrows = (self.landmarks[17:22] - np.array([0, 10], dtype=self.landmarks.dtype), 
                           self.landmarks[22:27] - np.array([0, 10], dtype=self.landmarks.dtype))

        parts = [r_eye, l_eye, nose, mouth, wider_mouth, higher_eyebrows]



        mergeds = []
        for item in parts:
            merged = np.concatenate(item)
            mergeds.append(merged)
        merged = np.concatenate(mergeds)
        cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask

    def display_landmarks(self):
        image = self.face.copy()
        for idx, point in enumerate(self.landmarks):
            x, y = point
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw a small green dot
            cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Label the dot
        
        rand_idx = np.random.randint(0, 1000)
        cv2.imwrite(f'ldm_points/{rand_idx}.png', image)


class extended(Mask):  # pylint: disable=invalid-name
    """ Extended mask
        Based on components mask. Attempts to extend the eyebrow points up the forehead
    """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        landmarks = self.landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        l_eye = (landmarks[22:27], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class facehull(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 255.0, lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask