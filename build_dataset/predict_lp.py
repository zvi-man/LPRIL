from typing import Callable
import pandas as pd


class TrackletLPRPredictorException(Exception):
    pass

# TODO:
#  1. write code to read the json file
#  2. write function to predict tracklet LP
#  3. write code to evaluate prediction

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
    pass


def predict_most_common_lp(single_tracklet_table: pd.DataFrame) -> str:
    pass


def predict_median_lp(single_tracklet_table: pd.DataFrame) -> str:
    pass


def predict_lp_top_confidence(single_tracklet_table: pd.DataFrame) -> str:
    pass


def predict_lp_size_and_confidence(single_tracklet_table: pd.DataFrame) -> str:
    pass


class TrackletLPRPredictor(object):
    @classmethod
    def evaluate_prediction_on_table(cls, pred: Callable, table: pd.DataFrame):
        tracklet_ids = cls.get_all_possible_tracklet_ids(table)
        for tracklet_id in tracklet_ids:
            tracklet_table = cls.get_tracklet_id_table(table, tracklet_id)
            lpn = pred(tracklet_table)
            gt_lpn = cls.get_tracklet_gt_lpn(tracklet_table)




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
