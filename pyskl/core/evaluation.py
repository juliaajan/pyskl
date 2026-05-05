# Copyright (c) OpenMMLab. All rights reserved.
# EarlyStoppingHook based on open PR contirbuted by 24hours:  https://github.com/open-mmlab/mmcv/pull/1602

import warnings
from numbers import Number
from typing import Callable, Dict
import torch


import numpy as np
from mmcv.runner import DistEvalHook as BasicDistEvalHook, Hook

class EarlyStoppingHook(Hook):
    """Stop training when a monitored metric has stopped improving.

    Args:
        monitor(str): The metric to monitor for improvement, e.g. 'top1_acc', 'loss'.
        phase(str): Determines whether early stopping should be performed based on train or validation metrics. Default is 'train'.
        min_delta(float): The minimum change in the monitored metric to qualify as an improvement. Default is 0.001.
        patience(int): Number of epochs to wait when no improvement is observed before stopping the training. Default is 3.
        mode(str): The comparison rule to determine if the monitored metric has improved, options are 'min' and 'max'.
            In 'min' mode, training will be stopped when the metric has stopped decreasing, and in 'max' mode, it will be stopped when the metric has stopped increasing.
        max_epochs(int): Maximum number of epochs to run before stopping the training regardless of improvement. Default is None, which means no maximum limit.

        To use, add the following configuration to your config file and adapt as needed:
            early_stopping = dict(
                monitor='top1_acc',
                phase='val',
                patience=3,
                min_delta=0.01,
                mode='max',
                max_epochs= 120)
   
     """

    mode_dict = {'min': np.less, 'max': np.greater}
    direction_dict = {'min': 'decrease', 'max': 'increase'}
    monitor_dict = {'top1_acc', 'loss'}

    def __init__(self, monitor: str, phase: str='val', min_delta: float=0.001, patience: int=3, mode: str='max', max_epochs: int=None):
        
        if monitor not in self.monitor_dict:
            raise ValueError(f"Monitoring metric must be one of {', '.join(self.monitor_dict.keys())}, but is {monitor}")
        
        if phase not in ['train', 'val']:
            raise ValueError(f"Phase must be one of 'train' and 'val', but is {phase}")
        
        if patience < 0:
            raise ValueError("Patience must be >= 0")
        
        if mode not in self.mode_dict:
            raise ValueError(f"Mode must be one of {', '.join(self.mode_dict.keys())}, but is {mode}")

        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be > 0 or None")
        
        self.monitor = monitor
        self.phase = phase
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.max_epochs = max_epochs
        self.wait_count = 0 #counter to check for how many epochs no improvement has been observed, to compare to patience
       
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf 

    def before_run(self, runner):
        if runner.meta is None:
            warnings.warn('runner.meta is None. Creating an empty one.')
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())
        self.wait_count = runner.meta['hook_msgs'].get('wait_count',
                                                       self.wait_count)
        self.best_score = runner.meta['hook_msgs'].get('early_stop_best_score',
                                                       self.best_score)

    def before_train_epoch(self, runner):        
        runner.log_buffer.clear()
            
    def after_train_epoch(self, runner):
        runner.log_buffer.average()
        #runner.logger.info("############## LOSS (aus log buffer)")
        #runner.logger.info(f"Loss : {runner.log_buffer.output['loss']}")
        runner.logger.info("############## LOSS")
        #runner.logger.info("############## LOSS (aus altem buffer über hook.msg)")
        #runner.logger.info(f"Loss : {runner.meta.get('hook_msgs', {}).get('last_eval_res', {})}")
        runner.logger.info(runner.log_buffer.output['loss'])
        #TODO: dafür sorgen dass logbuffer nicht mehr überschreibe und das hier weiterhin verwenden kann
        self._run_early_stopping_check(runner, runner.log_buffer.output)


   
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def get_mode(self, runner):
        return runner.mode

    def _get_current_score(self, runner, logs: Dict[str, float]):
        #workaround to get the top1_acc score in the validation phase as after_val_epoch is not used. Instead, validation is performed at the end of the train phase by DistEvalHook
        monitor = None
        if self.phase == 'val' and self.monitor == 'top1_acc':
            val_logs = runner.meta.get('hook_msgs', {}).get('last_eval_res', {})
            monitor = val_logs.get(self.monitor)
        elif self.phase =='val' and self.monitor == 'loss':
            val_logs = runner.meta.get('hook_msgs', {}).get('last_val_loss', {})
            monitor = val_logs.get('val_loss')

            runner.logger.info("##### Validation loss von mir geholt")
            runner.logger.info(monitor) #sollte nicht gleicher wert sein wie unter "LOSS (aus log buffer)"", den bisher verwende, sondern anderer wert

        else:
            monitor = logs.get(self.monitor).squeeze()

        if monitor is None:
            raise RuntimeError(f'Early stopping metric was set to {self.monitor}, which is not available in the logs.')
        
        runner.logger.info(f"Phase is set to {self.phase}, current {self.monitor} for last {self.phase} epoch is: {monitor}")        
        return monitor

    def _run_early_stopping_check(self, runner, logs: Dict[str, float]):
        runner.logger.info(f'Starting EarlyStoppingCheck. Current epoch: {runner.epoch +1}. Max epochs are currently set to: {runner._max_epochs}')

        #get the current score from the logs
        current_score = self._get_current_score(runner, logs)

        should_stop, reason = self._evaluate_stopping_criteria(current_score, runner)
        runner.meta['hook_msgs']['wait_count'] = self.wait_count
        runner.meta['hook_msgs']['early_stop_best_score'] = self.best_score
        
        #stop training:
        if should_stop:
            #stop training by setting the max_epochs in the runner to the current epoch (+1 because epochs are 0-based)
            runner._max_epochs = runner.epoch + 1
            runner.logger.info(f"Stopped training due to early stopping criteria. Reason: {reason}")
        else:
            runner.logger.info(f"Early stopping criteria not met. Reason: {reason} Training will be continued...")



    def _evaluate_stopping_criteria(self, current_score: float, runner):
        should_stop = False
        reason = None

        #if max_epochs is set, directly stop once the max amount of epochs is reached, regardless of improvement
        if self.max_epochs is not None and runner.epoch +1 >= self.max_epochs:
            should_stop = True
            reason= f'The maximum number of epochs is reached. Max_epochs is set to {self.max_epochs}, current epoch is {runner.epoch +1}.'
            return should_stop, reason

        #metric did increase (decrease), update best score
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            should_stop = False
            reason = f"Metric {self.monitor} {self.direction_dict[self.mode]}ed. Last best score was {self.best_score:.3f}, current score is {current_score:.3f} with min_delta {self.min_delta}. New best score set to {current_score:.3f}."
            self.best_score = current_score
            self.wait_count = 0
        #metric did not increase (decrease), check if patience is exceeded
        else: 
            self.wait_count += 1
            reason = f"Monitored metric {self.monitor} did not {self.direction_dict[self.mode]} in the last {self.wait_count} epochs, but patience of {self.patience} is not exceeded yet. Best score was {self.best_score:.3f}, current score is {current_score:.3f} with min_delta {self.min_delta}. "
            if self.wait_count >= self.patience:
                should_stop = True
                reason = f"Monitored metric {self.monitor} did not further {self.direction_dict[self.mode]} in the last {self.wait_count} epochs. Best score was {self.best_score:.3f}, current score is {current_score:.3f} with min_delta {self.min_delta}. Signaling Runner to stop."

        return should_stop, reason


class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)
    
    #add method to get validation losses, as these are not saved by default after validation
    def _compute_val_loss(self, runner):
        model = runner.model
        model.eval()

        old_output = dict(runner.log_buffer.output) if hasattr(runner.log_buffer, 'output') else {}
        #TODO: wieder raus
        #runner.logger.info("############## OLD OUTPUT (vor Berechnung val loss)")
        #runner.logger.info(old_output)
        #runner.logger.info(runner.log_buffer.output)

        runner.log_buffer.clear()
        runner.log_buffer.clear_output()

        last_num_samples = None

        for data in self.dataloader:
            with torch.no_grad():
                outputs = model.train_step(data, runner.optimizer)

            if isinstance(outputs, dict) and 'log_vars' in outputs:
                runner.log_buffer.update(outputs['log_vars'], outputs.get('num_samples', 1))
                last_num_samples = outputs.get('num_samples', 1)

        if last_num_samples is None:
            val_loss_logs = {}
        else:
            runner.log_buffer.average(last_num_samples)
            val_loss_logs = {
                f'val_{k}': float(v)
                for k, v in runner.log_buffer.output.items()
                if isinstance(v, Number)
            }

        runner.log_buffer.clear()
        runner.log_buffer.clear_output()
        runner.log_buffer.output.update(old_output)
        #TODO: wieder raus
        #runner.logger.info("##############  OUTPUT (nach Berechnung val loss)")
        #runner.logger.info(old_output)
        #runner.logger.info(runner.log_buffer.output)


        return val_loss_logs


    def evaluate(self, runner, results):
        key_score = super().evaluate(runner, results)
    
        if runner.meta is None:
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())

        #persist current validation metrics for EarlyStoppingHook
        eval_logs = {
            k: float(v)
            for k, v in runner.log_buffer.output.items()
            if isinstance(v, Number)
        }

        #get val loss logs
        #val_loss_logs = self._compute_val_loss(runner)

        runner.meta['hook_msgs']['last_eval_res'] = eval_logs
        #runner.meta['hook_msgs']['last_val_loss'] = val_loss_logs


        return key_score


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
