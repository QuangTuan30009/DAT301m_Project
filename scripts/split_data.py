from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data_raw/data_cleaned.csv")
OUTPUT_DIR = Path("data")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    logger.info("Loaded dataset with shape: %s", df.shape)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Validation shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)


if __name__ == "__main__":
    main()