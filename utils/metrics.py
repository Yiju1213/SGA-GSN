# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import numpy as np
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import os
from extensions.emd import emd_module as emd

class ShapeMetrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'EMDistance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd.emdModule(),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt, require_emd=False):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            if not require_emd and 'emd' in item['eval_func']:
                _values[i] = torch.tensor(0.).to(gt.device) # set to 0
            else:
                eval_func = eval(item['eval_func']) # find func
                _values[i] = eval_func(pred, gt) # compute

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    # @classmethod
    # def _get_f_score(cls, pred, gt, th=0.01):

    #     """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
    #     b = pred.size(0)
    #     device = pred.device
    #     assert pred.size(0) == gt.size(0)
    #     if b != 1:
    #         f_score_list = []
    #         for idx in range(b):
    #             f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
    #         return sum(f_score_list)/len(f_score_list)
    #     else:
    #         pred = cls._get_open3d_ptcloud(pred)
    #         gt = cls._get_open3d_ptcloud(gt)

    #         dist1 = pred.compute_point_cloud_distance(gt)
    #         dist2 = gt.compute_point_cloud_distance(pred)

    #         recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    #         precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    #         result = 2 * recall * precision / (recall + precision) if recall + precision else 0.
    #         result_tensor = torch.tensor(result).to(device)
    #         return result_tensor

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """
        F-score for non-standardized point clouds(e.g., not normalized to [0,1])
        """
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            bbox = gt.get_axis_aligned_bounding_box()
            scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            threshold = scale * th

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < threshold for d in dist2)) / float(len(dist2))
            precision = float(sum(d < threshold for d in dist1)) / float(len(dist1))
            result = 2 * recall * precision / (recall + precision) if recall + precision else 0.
            result_tensor = torch.tensor(result).to(device)
            return result_tensor

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt) * 1000

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[3]['eval_object']
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1000

    def __init__(self, metric_name, values):
        self._items = ShapeMetrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value

class GraspStabilityMetrics(object):
    ITEMS = [{
        'name': 'AvgAcc',
        'enabled': True,
        'eval_func': 'cls._get_avg_acc',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Precision',
        'enabled': True,
        'eval_func': 'cls._get_precision',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Recall',
        'enabled': True,
        'eval_func': 'cls._get_recall',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'F1',
        'enabled': True,
        'eval_func': 'cls._get_f1',
        'is_greater_better': True,
        'init_value': 0
    }]

    @classmethod
    def get(cls, logits, targets):
        _items = cls.items()
        _values = [0] * len(_items)

        # For binary classification regression: convert logits to predictions using sigmoid and threshold 0.5
        # Handle both [B, 1] and [B] shapes
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # [B, 1] -> [B]

        pred_probs = torch.sigmoid(logits)  # Convert logits to probabilities
        pred = (pred_probs > 0.5).long()   # Apply threshold 0.5 to get binary predictions

        # Ensure targets are in the correct format (should be float for BCEWithLogitsLoss)
        targets = targets.float()
        targets_long = targets.long()  # Convert to long for comparison

        correct = (pred == targets_long).float()
        avg_acc = correct.mean()

        pred_pos = (pred == 1)
        true_pos = (targets_long == 1)
        tp = ((pred_pos & true_pos).float()).sum()
        fp = ((pred_pos & (~true_pos)).float()).sum()
        fn = (((~pred_pos) & true_pos).float()).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Assign values in order
        for i, item in enumerate(_items):
            if item['name'] == 'AvgAcc':
                _values[i] = avg_acc
            elif item['name'] == 'Precision':
                _values[i] = precision
            elif item['name'] == 'Recall':
                _values[i] = recall
            elif item['name'] == 'F1':
                _values[i] = f1
        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_avg_acc(cls, logits, targets):
        pred = logits.argmax(dim=1)
        return (pred == targets.long()).float().mean()

    @classmethod
    def _get_precision(cls, logits, targets):
        pred = logits.argmax(dim=1)
        targets = targets.long()
        pred_pos = (pred == 1)
        true_pos = (targets == 1)
        tp = ((pred_pos & true_pos).float()).sum()
        fp = ((pred_pos & (~true_pos)).float()).sum()
        precision = tp / (tp + fp + 1e-8)
        return precision

    @classmethod
    def _get_recall(cls, logits, targets):
        pred = logits.argmax(dim=1)
        targets = targets.long()
        pred_pos = (pred == 1)
        true_pos = (targets == 1)
        tp = ((pred_pos & true_pos).float()).sum()
        fn = (((~pred_pos) & true_pos).float()).sum()
        recall = tp / (tp + fn + 1e-8)
        return recall

    @classmethod
    def _get_f1(cls, logits, targets):
        precision = cls._get_precision(logits, targets)
        recall = cls._get_recall(logits, targets)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    def __init__(self, metric_name, values):
        self._items = GraspStabilityMetrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    @property
    def F1(self):
        """Allow direct access to F1 value."""
        for i, item in enumerate(self._items):
            if item['name'] == 'F1':
                return self._values[i]
        raise AttributeError("F1 metric not found.")