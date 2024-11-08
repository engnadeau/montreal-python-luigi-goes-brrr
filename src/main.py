import pickle
from pathlib import Path

import luigi
import pandas as pd
from loguru import logger
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Data Ingestion
class DataIngestion(luigi.Task):
    def output(self):
        return luigi.LocalTarget(DATA_DIR / "raw_data.csv")

    def run(self):
        logger.info("Starting data ingestion step.")
        logger.info("Loading iris dataset...")
        iris = load_iris(as_frame=True)
        df = pd.concat([iris.data, iris.target.rename("target")], axis=1)

        logger.info("Saving data to CSV...")
        df.to_csv(self.output().path, index=False)
        logger.info(f"Data ingestion completed, saved to {self.output().path}")


# Step 2: Data Normalization
class DataNormalization(luigi.Task):
    def requires(self):
        # NOTE: Notice that we require the TASK not a particular file name
        # Separation of concerns: DataNormalization does not need to know where the data comes from
        return DataIngestion()

    def output(self):
        return luigi.LocalTarget(DATA_DIR / "normalized_data.csv")

    def run(self):
        logger.info("Starting data normalization step.")

        logger.info("Loading data...")
        df = pd.read_csv(self.input().path)

        logger.info("Normalizing data...")
        scaler = StandardScaler()
        features = df.columns[:-1]  # Exclude the target column
        df[features] = scaler.fit_transform(df[features])

        logger.info("Saving normalized data to CSV...")
        df.to_csv(self.output().path, index=False)
        logger.info(f"Data normalization completed, saved to {self.output().path}")


# Step 3: Train/Test Split and Labeling
class LabelTrainTestSplit(luigi.Task):
    # NOTE: We can pass parameters to the task allowing for more flexibility and configuration management
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return DataNormalization()

    def output(self):
        # NOTE: Each task should only have one output
        return luigi.LocalTarget(DATA_DIR / "labeled_data.csv")

    def run(self):
        logger.info(
            f"Starting train/test labeling step with test_size={self.test_size} and random_state={self.random_state}."
        )

        logger.info("Loading data...")
        df = pd.read_csv(self.input().path)
        train, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        logger.info("Labeling data as train or test...")
        train["split"] = "train"
        test["split"] = "test"

        logger.info("Saving labeled data to CSV...")
        combined_df = pd.concat([train, test])
        combined_df.to_csv(self.output().path, index=False)
        logger.info(f"Data labeling completed, saved to {self.output().path}")


# Step 4: Save Train Data
class SaveTrainData(luigi.Task):
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return LabelTrainTestSplit(
            test_size=self.test_size, random_state=self.random_state
        )

    def output(self):
        return luigi.LocalTarget(DATA_DIR / "train_data.csv")

    def run(self):
        logger.info("Saving train data.")

        logger.info("Loading labeled data...")
        df = pd.read_csv(self.input().path)
        train_df = df[df["split"] == "train"].drop(columns=["split"])

        logger.info("Saving train data to CSV...")
        train_df.to_csv(self.output().path, index=False)
        logger.info(f"Train data saved to {self.output().path}")


# Step 5: Save Test Data
class SaveTestData(luigi.Task):
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)

    def requires(self):
        return LabelTrainTestSplit(
            test_size=self.test_size, random_state=self.random_state
        )

    def output(self):
        return luigi.LocalTarget(DATA_DIR / "test_data.csv")

    def run(self):
        logger.info("Saving test data.")

        logger.info("Loading labeled data...")
        df = pd.read_csv(self.input().path)
        test_df = df[df["split"] == "test"].drop(columns=["split"])

        logger.info("Saving test data to CSV...")
        test_df.to_csv(self.output().path, index=False)
        logger.info(f"Test data saved to {self.output().path}")


# Step 6: Model Training
class ModelTraining(luigi.Task):
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)
    max_iter = luigi.IntParameter(default=200)

    def requires(self):
        return SaveTrainData(test_size=self.test_size, random_state=self.random_state)

    def output(self):
        return luigi.LocalTarget(MODEL_DIR / "model.pkl")

    def run(self):
        logger.info(f"Starting model training step with max_iter={self.max_iter}.")

        logger.info("Loading train data...")
        train_df = pd.read_csv(self.input().path)

        logger.info("Training model...")
        model = LogisticRegression(max_iter=self.max_iter)
        model.fit(train_df.iloc[:, :-1], train_df["target"])

        logger.info("Saving model...")
        with open(self.output().path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model training completed and saved to {self.output().path}")


# Step 7: Model Validation
class ModelValidation(luigi.Task):
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)
    max_iter = luigi.IntParameter(default=200)

    def requires(self):
        return {
            "model": ModelTraining(
                test_size=self.test_size,
                random_state=self.random_state,
                max_iter=self.max_iter,
            ),
            "test_data": SaveTestData(
                test_size=self.test_size, random_state=self.random_state
            ),
        }

    def output(self):
        return luigi.LocalTarget(RESULTS_DIR / "validation_result.txt")

    def run(self):
        logger.info("Starting model validation step.")

        logger.info("Loading test data...")
        test_df = pd.read_csv(self.input()["test_data"].path)

        logger.info("Loading model...")
        with open(self.input()["model"].path, "rb") as f:
            model = pickle.load(f)

        logger.info("Validating model...")
        predictions = model.predict(test_df.iloc[:, :-1])
        accuracy = accuracy_score(test_df["target"], predictions)
        logger.info(f"Model accuracy: {accuracy}")

        logger.info("Saving validation results...")
        with open(self.output().path, "w") as f:
            f.write(f"Model accuracy: {accuracy}\n")
        logger.info(f"Results saved to {self.output().path}")


# Wrapper Task: Run the Entire Pipeline
class RunPipeline(luigi.WrapperTask):
    test_size = luigi.FloatParameter(default=0.3)
    random_state = luigi.IntParameter(default=42)
    max_iter = luigi.IntParameter(default=200)

    def requires(self):
        yield ModelValidation(
            test_size=self.test_size,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )


# Run the pipeline
if __name__ == "__main__":
    luigi.build(
        [RunPipeline(test_size=0.25, random_state=1, max_iter=300)],
        local_scheduler=True,
    )
