from os.path import join

from src.scripts.bbox_data_preproc import augment_save_data_df, load_bboxes_train_val_df
from src.scripts.utils import output_dir

if __name__ == "__main__":
    size = (1024, 1024)

    train_df, val_df = load_bboxes_train_val_df()

    augment_save_data_df(train_df, join(output_dir, 'train'), size)
    augment_save_data_df(val_df, join(output_dir, 'val'), size)
