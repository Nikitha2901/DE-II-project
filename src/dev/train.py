import os
import time
import tempfile
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import (MinMaxScaler,
                                   FunctionTransformer,
                                   OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from dateutil import parser
import ray
from ray import cloudpickle
from ray.train import ScalingConfig, RunConfig, report, Checkpoint
from ray.train.trainer import BaseTrainer


# TODO: Refactor classes to seperate files in a directory
# Begin Pipeline Functions
class JSONNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_cols=None):
        self.datetime_cols = datetime_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.json_normalize(X)
        if self.datetime_cols:  # Also converts datetime cols
            for col in self.datetime_cols:
                df[col] = df[col].apply(lambda x: parser.parse(x).timestamp())
        return df


def enforce_str(df):  # can't use lambdas
    return df.astype(str)


ENFORCE_STR = FunctionTransformer(enforce_str)


def enforce_bin_na(df):  # can't use lambdas
    return df.replace({True: 1, False: 0}).fillna(0)


ENFORCE_BIN_NA = FunctionTransformer(enforce_bin_na)


class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(f'DEBUGT type: {type(X)}')
        print(f'DEBUGT shape: {X.shape}')
        return X

    def fit(self, X, y=None, **fit_params):
        print(f'DEBUGF type: {type(X)}')
        print(f'DEBUGF shape: {X.shape}')
        # pd.set_option('display.max_rows', 5000)
        # pd.set_option('display.max_columns', 5000)
        # pd.set_option('display.width', 150)
        # print(f'DEBUGF dtype: {X.dtypes}')
        return self

# END Pipeline functions

# Begin Trainer Functions


# Note: There is a built in XGBoost trainer, I just wanted to use
# the same process for all our Trainer classes and we don't
# need the features of their implementation.
# If we have time, we can come back and treat these pipelines
# like hyperparameters to optimize (as well as the literal
# hyperparameters of the models)
class XGTrainer(BaseTrainer):
    def setup(self):

        binary_cols = ['fork', 'has_issues', 'has_projects',
                       'has_downloads', 'has_wiki', 'has_pages',
                       'has_discussions', 'archived', 'disabled',
                       'allow_forking', 'is_template',
                       'web_commit_signoff_required']
        numeric_cols = ['watchers_count', 'created_at', 'updated_at',
                        'pushed_at', 'size', 'forks_count',
                        'open_issues_count', 'forks', 'score']

        topics_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('topic_encoder', OneHotEncoder(
                handle_unknown="ignore", min_frequency=2)
             ),
            # ('topic_pca', PCA(n_components=10, svd_solver="arpack")),
            ])

        categorial_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('one_hot', OneHotEncoder(handle_unknown="ignore")),
            ])

        preprocess = Pipeline([
            ('json_normalizer', JSONNormalizer(
                datetime_cols=['created_at', 'updated_at', 'pushed_at']
                )),
            ('ensure_binary_no_nas', ENFORCE_BIN_NA),
            ('col_transformer', ColumnTransformer([
                ('topic_transform', topics_pipeline, ['topics']),
                ('one_hot', categorial_pipeline,
                 ['language', 'visibility', 'license.spdx_id']
                 ),
                ('binary', 'passthrough', binary_cols),
                ('scaled', MinMaxScaler(), numeric_cols),
                ], remainder='drop')),
            ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocess),
            ('model', xgb.XGBRegressor(
                colsample_bytree=.5,
                max_depth=3,
                subsample=.7
                )),
            ])

    def training_loop(self):
        train_dataset = self.datasets["train"].take_all()
        test_dataset = self.datasets["test"].take_all()
        y_train = [entry['stargazers_count'] for entry in train_dataset]
        y_test = [entry['stargazers_count'] for entry in test_dataset]
        self.pipeline.fit(train_dataset, y_train)
        xgb_pred = self.pipeline.predict(test_dataset)
        xgb_r2 = r2_score(y_test, xgb_pred)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            with open(os.path.join(temp_checkpoint_dir, "model.pickle"), "wb+") as f:
                cloudpickle.dump(self.pipeline, f)
                report(
                        {"r2": xgb_r2},
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir)
                        )


class RegTrainer(BaseTrainer):
    def setup(self):

        binary_cols = ['fork', 'has_issues', 'has_projects',
                       'has_downloads', 'has_wiki', 'has_pages',
                       'has_discussions', 'archived', 'disabled',
                       'allow_forking', 'is_template',
                       'web_commit_signoff_required']
        numeric_cols = ['watchers_count', 'created_at', 'updated_at',
                        'pushed_at', 'size', 'forks_count',
                        'open_issues_count', 'forks', 'score']

        topics_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('topic_encoder', OneHotEncoder(
                handle_unknown="ignore", min_frequency=2)
             ),
            ])

        categorial_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('one_hot', OneHotEncoder(handle_unknown="ignore")),
            ])

        preprocess = Pipeline([
            ('json_normalizer', JSONNormalizer(
                datetime_cols=['created_at', 'updated_at', 'pushed_at']
                )),
            ('ensure_binary_no_nas', ENFORCE_BIN_NA),
            ('col_transformer', ColumnTransformer([
                ('topic_transform', topics_pipeline, ['topics']),
                ('one_hot', categorial_pipeline,
                 ['language', 'visibility', 'license.spdx_id']
                 ),
                ('binary', 'passthrough', binary_cols),
                ('scaled', MinMaxScaler(), numeric_cols),
                ], remainder='drop')),
            ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocess),
            ('model', LinearRegression()),
            ])

    def training_loop(self):
        train_dataset = self.datasets["train"].take_all()
        test_dataset = self.datasets["test"].take_all()
        y_train = [entry['stargazers_count'] for entry in train_dataset]
        y_test = [entry['stargazers_count'] for entry in test_dataset]
        self.pipeline.fit(train_dataset, y_train)
        pred = self.pipeline.predict(test_dataset)
        r2 = r2_score(y_test, pred)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            with open(os.path.join(temp_checkpoint_dir, "model.pickle"), "wb+") as f:
                cloudpickle.dump(self.pipeline, f)
                report(
                        {"r2": r2},
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir)
                        )


class RFTrainer(BaseTrainer):
    def setup(self):

        binary_cols = ['fork', 'has_issues', 'has_projects',
                       'has_downloads', 'has_wiki', 'has_pages',
                       'has_discussions', 'archived', 'disabled',
                       'allow_forking', 'is_template',
                       'web_commit_signoff_required']
        numeric_cols = ['watchers_count', 'created_at', 'updated_at',
                        'pushed_at', 'size', 'forks_count',
                        'open_issues_count', 'forks', 'score']

        topics_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('topic_encoder', OneHotEncoder(
                handle_unknown="ignore", min_frequency=2)
             ),
            # ('topic_pca', PCA(n_components=10, svd_solver="arpack")),
            ])

        categorial_pipeline = Pipeline([
            ('ensure_str', ENFORCE_STR),
            ('one_hot', OneHotEncoder(handle_unknown="ignore")),
            ])

        preprocess = Pipeline([
            ('json_normalizer', JSONNormalizer(
                datetime_cols=['created_at', 'updated_at', 'pushed_at']
                )),
            ('ensure_binary_no_nas', ENFORCE_BIN_NA),
            ('col_transformer', ColumnTransformer([
                ('topic_transform', topics_pipeline, ['topics']),
                ('one_hot', categorial_pipeline,
                 ['language', 'visibility', 'license.spdx_id']
                 ),
                ('binary', 'passthrough', binary_cols),
                ('scaled', MinMaxScaler(), numeric_cols),
                ], remainder='drop')),
            ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocess),
            ('model', RandomForestRegressor()),
            ])

    def training_loop(self):
        train_dataset = self.datasets["train"].take_all()
        test_dataset = self.datasets["test"].take_all()
        y_train = [entry['stargazers_count'] for entry in train_dataset]
        y_test = [entry['stargazers_count'] for entry in test_dataset]
        self.pipeline.fit(train_dataset, y_train)
        pred = self.pipeline.predict(test_dataset)
        r2 = r2_score(y_test, pred)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            with open(os.path.join(temp_checkpoint_dir, "model.pickle"), "wb+") as f:
                cloudpickle.dump(self.pipeline, f)
                report(
                        {"r2": r2},
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir)
                        )


# END Trainer functions

# Start Ray Train
config = ScalingConfig(num_workers=1, use_gpu=False) # num workers is why scaling ray-workers does nothing, its already fixed by the ray cluster increasing is not allowed by Base Trainer
ray.init(address="auto")
# Load data using ray data - cluster will share the distributed dataset
dataset = ray.data.read_json("./repositories.json").random_shuffle()
print("Resources:\n", ray.cluster_resources())
train_data, test_data = dataset.train_test_split(test_size=.2)

start_time = time.time()

xg_trainer = XGTrainer(
        datasets={"train": train_data, "test": test_data},
        scaling_config=config,
		run_config = RunConfig(
			storage_path="/app/shared",
			)
        )
reg_trainer = RegTrainer(
        datasets={"train": train_data, "test": test_data},
        scaling_config=config,
		run_config = RunConfig(
			storage_path="/app/shared",
			)
        )
rf_trainer = RFTrainer(
        datasets={"train": train_data, "test": test_data},
        scaling_config=config,
		run_config = RunConfig(
			storage_path="/app/shared",
			)
        )

xg_result = xg_trainer.fit()
reg_result = reg_trainer.fit()
rf_result = rf_trainer.fit()

print(f'XGB R2:\n{xg_result.metrics["r2"]}')
print(f'Reg R2:\n{reg_result.metrics["r2"]}')
print(f'RF R2:\n{rf_result.metrics["r2"]}')

end_time = time.time()

# Could handle all this and above with a loop and be more elegant about it
# bet welp, I am tired and just want this section done. Also, the best model
# would benefit from rerunning training with the whole dataset and/or using
# k-fold in the training loop but we can add that later if we are motivated
# and have time.
best_model_result = max(
        [xg_result, reg_result, rf_result],
        key=lambda x: x.metrics['r2']
        )

best_checkpoint = best_model_result.checkpoint

# Get best model
with best_checkpoint.as_directory() as checkpoint_dir:
    model_path = Path(checkpoint_dir).joinpath("model.pickle")
    assert model_path.exists()
    with open(model_path, "rb") as rf:
        MODEL = cloudpickle.load(rf)
        print(f'FINAL MODEL:\n{MODEL}')

# Save best model
with open("./shared/FINAL_MODEL.pickle", "wb+") as f:
    cloudpickle.dump(MODEL, f)
print("Model saved")
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
