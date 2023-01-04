from typing import Callable, Tuple

import pandas as pd


class TrackletLPRPredictorException(Exception):
    pass


# TODO:
#  1. write function to predict tracklet LP
#  2. write functions to filter table before prediction
#  3. run code to test combinations
#  4. plot precision recall curves for all prediction options

# Constants
UNLABELED_LPN = "AAAAAA"
LP_PREDICTION_UNCERTAIN = "Uncertain"
TRACKLET_ID_COL = "ID"
WIDTH_COL = "width"
HEIGHT_COL = "height"
SINGLE_IMG_LP_COL = "single_img_lp"
SINGLE_IMG_LP_CONFIDENCE_COL = "single_img_lp_confidence"
PREDICTION_LP_COL = "prediction_lp"
LABELED_LP_COL = "labeled_lp"


def predict_lp_of_biggest_car(single_tracklet_table: pd.DataFrame) -> str:
    new_col_name = "CAR_PIXEL_COUNT"
    single_tracklet_table[new_col_name] = single_tracklet_table[HEIGHT_COL] * single_tracklet_table[WIDTH_COL]
    max_line = single_tracklet_table[new_col_name].idxmax()
    return single_tracklet_table[SINGLE_IMG_LP_COL][max_line]


def predict_most_common_lp(single_tracklet_table: pd.DataFrame) -> str:
    value_counts = single_tracklet_table[SINGLE_IMG_LP_COL].value_counts(sort=True)
    if value_counts[0] == value_counts[1]:
        return LP_PREDICTION_UNCERTAIN
    return value_counts.index[0]


def predict_most_common_lp_conditioned(single_tracklet_table: pd.DataFrame) -> str:
    value_counts = single_tracklet_table[SINGLE_IMG_LP_COL].value_counts(sort=True)
    if value_counts[0] == value_counts[1]:
        return LP_PREDICTION_UNCERTAIN
    if value_counts[0] / single_tracklet_table.shape[0] < 0.7:
        return LP_PREDICTION_UNCERTAIN
    return value_counts.index[0]


def predict_lp_top_confidence(single_tracklet_table: pd.DataFrame) -> str:
    pass


def predict_lp_size_and_confidence(single_tracklet_table: pd.DataFrame) -> str:
    pass


class TrackletLPRPredictor(object):
    @classmethod
    def calc_prediction_accuracy(cls, pred: Callable, table: pd.DataFrame) -> Tuple[float, float]:
        tracklet_ids = cls.get_all_possible_tracklet_ids(table)
        num_evaluated_tracklets = 0
        num_correct_tracklets = 0
        num_gt_tracklets = 0
        for tracklet_id in tracklet_ids:
            tracklet_table = cls.get_tracklet_id_table(table, tracklet_id)
            pred_lpn = pred(tracklet_table)
            gt_lpn = cls.get_tracklet_gt_lpn(tracklet_table)
            if gt_lpn != UNLABELED_LPN:
                num_gt_tracklets += 1
                if pred_lpn != LP_PREDICTION_UNCERTAIN:
                    num_evaluated_tracklets += 1
                    if pred_lpn == gt_lpn:
                        num_correct_tracklets += 1

        if num_gt_tracklets == 0:
            raise TrackletLPRPredictorException("There are no tagged tracklets - could not evaluate")
        prediction_precision = num_correct_tracklets / num_evaluated_tracklets
        prediction_recall = num_correct_tracklets / num_gt_tracklets
        return prediction_precision, prediction_recall

    @staticmethod
    def get_all_possible_tracklet_ids(table: pd.DataFrame):
        return set(table["ID"].to_list())

    @staticmethod
    def get_tracklet_id_table(table: pd.DataFrame, tracklet_id: int):
        return table[table[TRACKLET_ID_COL] == tracklet_id]

    @staticmethod
    def get_tracklet_gt_lpn(tracklet_table: pd.DataFrame) -> str:
        if LABELED_LP_COL not in tracklet_table.columns:
            raise TrackletLPRPredictorException(f"LP Table does not contain {LABELED_LP_COL} col")
        gt_lpn_set = set(tracklet_table[LABELED_LP_COL].to_list())
        if UNLABELED_LPN in gt_lpn_set:
            gt_lpn_set.remove(UNLABELED_LPN)
        if len(gt_lpn_set) != 1:
            tracklet_num = set(tracklet_table[TRACKLET_ID_COL].to_list())
            raise TrackletLPRPredictorException(f"There must be only one ground truth LPN per tracklet. "
                                                f"got  {gt_lpn_set} for tracklet {tracklet_num}")
        return gt_lpn_set.pop()


def load_lp_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


if __name__ == '__main__':
    lp_table_csv_path = "build_dataset/lp_table.csv"
    lp_table = load_lp_table(lp_table_csv_path)
    precision, recall = TrackletLPRPredictor.calc_prediction_accuracy(predict_lp_of_biggest_car, lp_table)
    print(f"biggest_car precision, recall: {precision}, {recall}")
    precision, recall = TrackletLPRPredictor.calc_prediction_accuracy(predict_most_common_lp, lp_table)
    print(f"most_common_lp precision, recall: {precision}, {recall}")
    precision, recall = TrackletLPRPredictor.calc_prediction_accuracy(predict_most_common_lp_conditioned, lp_table)
    print(f"most_common_lp_conditioned precision, recall: {precision}, {recall}")
    # biggest_car_acc = TrackletLPRPredictor.calc_prediction_accuracy(predict_lp_of_biggest_car, lp_table)
    # biggest_car_acc = TrackletLPRPredictor.calc_prediction_accuracy(predict_lp_of_biggest_car, lp_table)
