from typing import List, Union, Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch 
import json

from evalutils.scorers import score_detection
from torchmetrics.detection import MeanAveragePrecision


def _F1_core_balltree(
    annos: np.ndarray, 
    boxes: np.ndarray, 
    scores: np.ndarray, 
    det_thresh: float, 
    radius: int = 25) -> Tuple[float, int, int, int]:
    """Computes F1, TP, FP, FN scores for a given set of annotations and detections.

    Args:
        annos (np.ndarray): array of bounding box coordinates in the format [xmin,ymin,xmax,ymax].
        boxes (np.ndarray): predicted bounding boxes in the format [xmin,ymin,xmax,ymax].
        scores (np.ndarray): predicted scores.
        det_thresh (float): detection threshold.
        radius (int, optional): radius of kd-tree query. Defaults to 25.

    Returns:
        Tuple[float, int, int, int]: f1, tp, fp, fn.
    """
    # filter detections by threshold 
    keep = scores > det_thresh
    boxes = boxes[keep]

    # compute scores
    scores = score_detection(ground_truth=annos, predictions=boxes, radius=radius)

    # extract scores
    TP = scores.true_positives
    FP = scores.false_positives
    FN = scores.false_negatives

    eps = 1e-12

    # compute F1
    F1 = (2 * TP) / ((2 * TP) + FP + FN + eps)

    return F1, TP, FP, FN



def optimize_threshold(
    dataset: pd.DataFrame,
    preds: Dict[str, np.ndarray],
    min_thresh: float = 0.3,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Optimizes the detection threshold based on F1-score over a range of thresholds.

    Args:
        dataset (pd.DataFrame): Dataset for optimization.
        preds (Dict[str, np.ndarray]): Predictions to evaluate.
        min_thresh (float, optional): Minimum threshold. Defaults to 0.3.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: Best threshold, best F1-score, all F1-scores, all thresholds
    """

    F1dict = dict()

    TPd, FPd, FNd, F1d = dict(), dict(), dict(), dict()
    thresholds = np.round(np.arange(min_thresh,0.99,0.001), decimals=3)

    for resfile in preds:
        boxes = np.array(preds[resfile]['boxes'])

        TP, FP, FN = 0,0,0
        TPd[resfile] = list()
        FPd[resfile] = list()
        FNd[resfile] = list()
        F1d[resfile] = list()

        if (boxes.shape[0]>0):
            score = preds[resfile]['scores']
            
            # get annotations  
            annos = dataset[['xmin', 'ymin', 'xmax', 'ymax']].loc[dataset.filename == resfile].values

            for det_thres in thresholds:
                F1,TP,FP,FN = _F1_core_balltree(annos, boxes, score, det_thres)
                TPd[resfile] += [TP]
                FPd[resfile] += [FP]
                FNd[resfile] += [FN]
                F1d[resfile] += [F1]
        else:
            for det_thres in thresholds:
                TPd[resfile] += [0]
                FPd[resfile] += [0]
                FNd[resfile] += [0]
                F1d[resfile] += [0]
            F1 = 0
            
        F1dict[resfile]=F1

    allTP = np.zeros(len(thresholds))
    allFP = np.zeros(len(thresholds))
    allFN = np.zeros(len(thresholds))
    allF1 = np.zeros(len(thresholds))
    allF1M = np.zeros(len(thresholds))

    for k in range(len(thresholds)):
        allTP[k] = np.sum([TPd[x][k] for x in preds])
        allFP[k] = np.sum([FPd[x][k] for x in preds])
        allFN[k] = np.sum([FNd[x][k] for x in preds])
        allF1[k] = 2*allTP[k] / (2*allTP[k] + allFP[k] + allFN[k])
        allF1M[k] = np.mean([F1d[x][k] for x in preds])

    return thresholds[np.argmax(allF1)], np.max(allF1), allF1, thresholds






class MIDOGEvaluation:
    def __init__(
            self, 
            gt_file: Union[str, pd.DataFrame],
            preds: Dict[str, Dict[str, Any]],
            output_file: str,
            det_thresh: float,
            split: str = 'test',
            bbox_size: int = 50,
            radius: int = 25
    ) -> None:
        self.gt_file = gt_file
        self.preds = preds
        self.output_file = output_file
        self.det_thresh = det_thresh
        self.split = split
        self.bbox_size = bbox_size
        self.radius = radius

        self.load_gt()

        self.ap = MeanAveragePrecision(
            iou_thresholds=[0.5],
            class_metrics=False,
            iou_type='bbox',
            box_format='xyxy',
            max_detection_thresholds=[1, 100, 10000],
            backend='faster_coco_eval'
        )
        self.per_tumor_ap = {tumor: MeanAveragePrecision(
            iou_thresholds=[0.5],
            class_metrics=False,
            iou_type='bbox',
            box_format='xyxy',
            max_detection_thresholds=[1, 100, 10000],
            backend='faster_coco_eval'
        ) for tumor in self.tumor_cases}

    
    def load_gt(self) -> None:
        """Load ground truth annotations and case to tumor dictionary."""
        if isinstance(self.gt_file, str):
            dataset = pd.read_csv(self.gt_file)
        elif isinstance(self.gt_file, pd.DataFrame):
            dataset = self.gt_file
        else:
            raise TypeError('Dataset must be either str or pd.Dataframe. Got {}'.format(type(self.gt_file)))
        
        dataset = dataset.query('label == 1 and split == @self.split')

        gt = {}
        case_to_tumor = {}
        for _case in dataset.filename.unique():
            coords = dataset.query('filename == @_case')[['x', 'y']].to_numpy()
            tumor = dataset.query('filename == @_case')['tumortype'].unique().item()
            gt[_case] = coords
            case_to_tumor[_case] = tumor

        self.gt = gt
        self.case_to_tumor = case_to_tumor
        self.tumor_cases = dataset['tumortype'].unique()



    @property
    def _metrics(self) -> Dict:
        """Returns the calculated case and aggregate results"""
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        }  


    def score(self) -> None:
        """Computes case specific and aggregated results"""
        
        # init case specific results 
        self._case_results = {}

        for idx, _case in enumerate(self.gt.keys()):
            if _case not in self.preds:
                print('Warning: No prediction for file: ', _case)
                continue

            # get case predictions
            case_preds = self.preds[_case]

            preds_dict = [
                {'boxes': torch.tensor(case_preds['boxes'], dtype=torch.float),
                 'scores': torch.tensor(case_preds['scores'], dtype=torch.float),
                 'labels': torch.tensor(case_preds['labels'], dtype=torch.int)}
            ]

            # get case targets 
            case_targets = self.gt[_case]
            
            bbox_radius = self.bbox_size / 2.

            target_dict = [
                {'boxes': torch.tensor([[x-bbox_radius, y-bbox_radius, x+bbox_radius, y+bbox_radius] for x, y in case_targets], dtype=torch.float),
                 'labels': torch.tensor([1,]*len(case_targets), dtype=torch.int)}
            ]

            # update ap metrics
            self.ap.update(preds_dict, target_dict)
            self.per_tumor_ap[self.case_to_tumor[_case]].update(preds_dict, target_dict)


            # compute scores
            F1, tp, fp, fn = _F1_core_balltree(
                annos=[[x-bbox_radius, y-bbox_radius, x+bbox_radius, y+bbox_radius] for x, y in case_targets],
                boxes=case_preds['boxes'],
                scores=case_preds['scores'],
                det_thresh=self.det_thresh,
                radius=self.radius
            )

            self._case_results[_case] = {'tp': tp, 'fp': fp, 'fn': fn}

        # compute aggregate results 
        self._aggregate_results = self.score_aggregates()


    def score_aggregates(self) -> Dict[str, float]:

        # init per tumor scores
        per_tumor = {tumor: {'tp': 0, 'fp': 0, 'fn': 0} for tumor in self.tumor_cases}

        # accumulate case specific scores
        tp, fp, fn = 0, 0, 0
        for case, scores in self._case_results.items():
            tp += scores['tp']
            fp += scores['fp']
            fn += scores['fn']

            per_tumor[self.case_to_tumor[case]]['tp']  += scores['tp']
            per_tumor[self.case_to_tumor[case]]['fp']  += scores['fp']
            per_tumor[self.case_to_tumor[case]]['fn']  += scores['fn']

        # init aggregated resutls
        aggregate_results = {}

        eps = 1e-12

        aggregate_results["precision"] = tp / (tp + fp + eps)
        aggregate_results["recall"] = tp / (tp + fn + eps)
        aggregate_results["f1_score"] = (2 * tp) / ((2 * tp) + fp + fn + eps)

        metric_values = self.ap.compute()
        aggregate_results["AP"] = metric_values['map_50'].tolist()

        # compute tumor specific restuls 
        for tumor in per_tumor:
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_precision"] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fp'] + eps)
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_recall"] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fn'] + eps)
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_f1"] = (2 * per_tumor[tumor]['tp']) / ((2 * per_tumor[tumor]['tp']) + per_tumor[tumor]['fp'] + per_tumor[tumor]['fn'] + eps)

            per_tumor_metric_values = self.per_tumor_ap[tumor].compute()
            aggregate_results[f"tumor_{tumor.replace(' ', '')}_AP"] = per_tumor_metric_values['map_50'].tolist()

        return aggregate_results


    def save(self):
        with open(self.output_file, "w") as f:
                    f.write(json.dumps(self._metrics, cls=NpEncoder))  
    

    def evaluate(self, verbose: bool=False):
        self.score()
        self.save()

        
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
    def encode(self, obj):
        if isinstance(obj, dict):
            return super(NpEncoder, self).encode(self._convert_keys(obj))
        return super(NpEncoder, self).encode(obj)

    def _convert_keys(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if isinstance(key, np.integer):
                    new_key = int(key)
                else:
                    new_key = key
                if isinstance(value, dict):
                    new_dict[new_key] = self._convert_keys(value)
                else:
                    new_dict[new_key] = value
            return new_dict
        return obj