import itertools
from typing import Callable, Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from KMUtils.GeneralUtils.plot_utils import PlotUtils


class Img2TrackletRecognitionPredictorException(Exception):
    pass


# TODO:
#  1. write functions to add size filter

# Constants
WIDTH_COL = "width"
HEIGHT_COL = "height"
SINGLE_IMG_LP_COL = "single_img_lp"
SINGLE_IMG_LP_CONFIDENCE_COL = "single_img_lp_confidence"
PREDICTION_LP_COL = "prediction_lp"
PRECISION_COL = "precision"
RECALL_COL = "recall"
COMMON_LP_THRESH_HOLD = 0.7


def predict_tracklet_by_img_size(single_tracklet_table: pd.DataFrame) -> str:
    new_col_name = "CAR_PIXEL_COUNT"
    single_tracklet_table[new_col_name] = single_tracklet_table[HEIGHT_COL] * single_tracklet_table[WIDTH_COL]
    max_line = single_tracklet_table[new_col_name].idxmax()
    return single_tracklet_table[SINGLE_IMG_LP_COL][max_line]


def predict_tracklet_by_most_common(single_tracklet_table: pd.DataFrame,
                                    common_lp_thresh_hold: float = COMMON_LP_THRESH_HOLD) -> str:
    value_counts = single_tracklet_table[SINGLE_IMG_LP_COL].value_counts(sort=True)
    if len(value_counts) > 1 and value_counts[0] == value_counts[1]:
        return ''
    if value_counts[0] < common_lp_thresh_hold:
        return ''
    return value_counts.index[0]


def predict_lp_top_confidence(single_tracklet_table: pd.DataFrame,
                              confidence_thresh_hold: float = 0.5) -> str:
    max_line = single_tracklet_table[SINGLE_IMG_LP_CONFIDENCE_COL].idxmax()
    confidence = single_tracklet_table[SINGLE_IMG_LP_CONFIDENCE_COL].max()
    if confidence < confidence_thresh_hold:
        return ''
    else:
        return single_tracklet_table[SINGLE_IMG_LP_COL][max_line]


def predict_lp_size_conf_comm(single_tracklet_table: pd.DataFrame,
                              confidence_thresh_hold: float,
                              min_car_area: int) -> str:
    car_area_col = "car_pixel_count"
    s_t_table = single_tracklet_table  # give shorter name
    s_t_table[car_area_col] = s_t_table[HEIGHT_COL] * s_t_table[WIDTH_COL]
    sub_table = s_t_table.loc[(s_t_table[car_area_col] > min_car_area) &
                              (s_t_table[SINGLE_IMG_LP_CONFIDENCE_COL] > confidence_thresh_hold)]
    # select most common
    value_counts = single_tracklet_table[SINGLE_IMG_LP_COL].value_counts(sort=True)
    if len(value_counts) == 0:
        return ''
    if len(value_counts) > 1 and value_counts[0] == value_counts[1]:
        return ''
    return value_counts.index[0]


class Img2TrackletRecognitionPredictor(object):
    TRACKLET_ID_COL = "ID"
    TRACKLET_GT_COL = "gt"
    UNLABELED_STR = "AAAAAA"
    PREDICTION_UNCERTAIN = ''

    @classmethod
    def calc_prediction_precision_recall_point(cls, pred: Callable,
                                               table: pd.DataFrame) -> Tuple[float, float]:
        tracklet_ids = cls.get_all_possible_tracklet_ids(table)
        num_evaluated_tracklets = 0
        num_correct_tracklets = 0
        num_gt_tracklets = 0
        for tracklet_id in tracklet_ids:
            tracklet_table = cls.get_tracklet_id_table(table, tracklet_id)
            pred_label = pred(tracklet_table)
            gt_label = cls.get_tracklet_gt_label(tracklet_table)
            if gt_label != cls.UNLABELED_STR:
                num_gt_tracklets += 1
                if pred_label != cls.PREDICTION_UNCERTAIN:
                    num_evaluated_tracklets += 1
                    if pred_label == gt_label:
                        num_correct_tracklets += 1

        if num_gt_tracklets == 0:
            raise Img2TrackletRecognitionPredictorException("There are no tagged tracklets - could not evaluate")
        if num_evaluated_tracklets == 0:
            return 1.0, 0.0  # Filtered All Data - Precision = 100%, Recall 0%
        prediction_precision = num_correct_tracklets / num_evaluated_tracklets
        prediction_recall = num_correct_tracklets / num_gt_tracklets
        return prediction_precision, prediction_recall

    @classmethod
    def calc_prediction_precision_recall_curve(cls, pred: Callable,
                                               table: pd.DataFrame,
                                               threshold_list: List[float]) -> Tuple[List[float], List[float]]:
        precision_results = []
        recall_results = []
        for threshold in threshold_list:
            def single_point_pred(my_table: pd.DataFrame) -> str:
                return pred(my_table, threshold)

            point_precision, point_recall = cls.calc_prediction_precision_recall_point(single_point_pred, table)
            precision_results.append(point_precision)
            recall_results.append(point_recall)
        return precision_results, recall_results

    @classmethod
    def calc_prediction_precision_recall_matrix(cls, pred: Callable,
                                                table: pd.DataFrame,
                                                threshold_dict: Dict[str, List[float]]) -> pd.DataFrame:
        parameter_names_list = threshold_dict.keys()
        parameter_values = [threshold_dict[param] for param in parameter_names_list]
        df_results = pd.DataFrame(list(itertools.product(*parameter_values)),
                                  columns=parameter_names_list)
        df_results[PRECISION_COL] = 0.0
        df_results[RECALL_COL] = 0.0
        for index, row in df_results.iterrows():
            parameter_val_dict = {param: row[param] for param in parameter_names_list}

            def single_point_pred(my_table: pd.DataFrame) -> str:
                return pred(my_table, **parameter_val_dict)
            print(parameter_val_dict)
            point_precision, point_recall = cls.calc_prediction_precision_recall_point(single_point_pred, table)
            if point_precision != 0.0 or point_recall != 0.0:
                print(f"{point_precision}, {point_recall}")
            df_results.at[index, PRECISION_COL] = point_precision
            df_results.at[index, RECALL_COL] = point_recall
        return df_results

    @classmethod
    def get_all_possible_tracklet_ids(cls, table: pd.DataFrame):
        return set(table[cls.TRACKLET_ID_COL].to_list())

    @classmethod
    def get_tracklet_id_table(cls, table: pd.DataFrame, tracklet_id: int):
        return table[table[cls.TRACKLET_ID_COL] == tracklet_id]

    @classmethod
    def get_tracklet_gt_label(cls, tracklet_table: pd.DataFrame) -> str:
        if cls.TRACKLET_GT_COL not in tracklet_table.columns:
            raise Img2TrackletRecognitionPredictorException(f"Table does not contain {cls.TRACKLET_GT_COL} col")
        gt_label_set = set(tracklet_table[cls.TRACKLET_GT_COL].to_list())
        if cls.UNLABELED_STR in gt_label_set:
            gt_label_set.remove(cls.UNLABELED_STR)
        if len(gt_label_set) != 1:
            tracklet_num = set(tracklet_table[cls.TRACKLET_ID_COL].to_list())
            raise Img2TrackletRecognitionPredictorException(f"There must be only one ground truth label per tracklet. "
                                                            f"got  {gt_label_set} for tracklet {tracklet_num}")
        return gt_label_set.pop()


if __name__ == '__main__':
    lp_table_csv_path = "build_dataset/lp_table.csv"
    lp_table = pd.read_csv(lp_table_csv_path)

    pr_dict = {}
    precision, recall = Img2TrackletRecognitionPredictor.calc_prediction_precision_recall_point(
        predict_tracklet_by_img_size, lp_table)
    print(f"biggest_car precision, recall: {precision}, {recall}")
    pr_dict["biggest_car"] = (precision, recall)

    precision, recall = Img2TrackletRecognitionPredictor.calc_prediction_precision_recall_curve(
        predict_tracklet_by_most_common, lp_table, list(range(5)))
    pr_dict["most_common_lp"] = (precision, recall)
    print(f"most_common_lp precision, recall: {precision}, {recall}")
    print(f"most_common_lp AP: {auc(recall, precision)}")

    precision, recall = Img2TrackletRecognitionPredictor.calc_prediction_precision_recall_curve(
        predict_lp_top_confidence, lp_table, list(np.linspace(0, 1, 10)))
    pr_dict["predict_lp_top_confidence"] = (precision, recall)
    print(f"predict_lp_top_confidence precision, recall: {precision}, {recall}")
    print(f"predict_lp_top_confidence AP: {auc(recall, precision)}")

    threshold_dict = {'confidence_thresh_hold': list(np.linspace(0, 1, 10)),
                      'min_car_area': list(range(100, 1000, 100))}
    df = Img2TrackletRecognitionPredictor.calc_prediction_precision_recall_matrix(predict_lp_size_conf_comm,
                                                                                  lp_table,
                                                                                  threshold_dict)

    print(df)
    PlotUtils.plot_multiple_precision_recall_curves(pr_dict)
