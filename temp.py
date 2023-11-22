from typing import Tuple, List

import pandas as pd
import numpy as np
import os.path as osp

# Constants
BACK_DIRECTION = "Back"


def get_emb_by_direction(embeddings: np.ndarray, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    indexes_back = []
    indexes_front = []
    for i, file_path in enumerate(file_paths):
        father_dir = osp.basename(osp.dirname(file_path))
        if father_dir == BACK_DIRECTION:
            indexes_back.append(i)
        else:
            indexes_front.append(i)
    return embeddings[indexes_front], embeddings[indexes_back]


def combine_df_scores(df: pd.DataFrame, columns_to_merge: List[str], new_col_name: str) -> pd.DataFrame:
    df[new_col_name] = df.loc[:, columns_to_merge].prod(axis=1)
    return df


def example_split_embeddings_by_path():
    emb = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    file_paths = ["aaa/1.jpg", "aaa/2.jpg", "aaa/3.jpg", "aaa/4.jpg"]
    front_emb, back_emb = get_emb_by_direction(emb, file_paths)
    print(f"front embeddings: {front_emb}")
    print(f"back embeddings: {back_emb}")


def example_combine_df_scores():
    df = pd.DataFrame({'score': [0.1, 0.2, 0.3, 0.4], 'color_true': [True, True, False, False],
                       'maker_true': [True, False, True, False]})
    combine_df_scores(df, ['score', 'color_true', 'maker_true'], 'RCM')
    combine_df_scores(df, ['score', 'color_true'], 'RC')
    combine_df_scores(df, ['score', 'maker_true'], 'RM')
    print(df)


if __name__ == '__main__':
    example_combine_df_scores()
    # example_split_embeddings_by_path()
