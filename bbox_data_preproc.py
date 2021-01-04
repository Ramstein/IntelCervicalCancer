from os.path import join

from src.scripts.bbox_data_preproc import augment_save_data_df, load_bboxes_train_val_df
from src.scripts.utils import aug_output_dir, img_size

if __name__ == "__main__":
    train_df, val_df = load_bboxes_train_val_df()

    augment_save_data_df(train_df, join(aug_output_dir, 'train'), img_size)
    augment_save_data_df(val_df, join(aug_output_dir, 'val'), img_size)
