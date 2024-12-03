from catboost import CatBoostClassifier, Pool
from loguru import logger
from sklearn.metrics import classification_report, precision_recall_fscore_support

from recsys.config import settings


class RankingModelFactory:
    @classmethod
    def build(cls) -> CatBoostClassifier:
        return CatBoostClassifier(
            learning_rate=settings.RANKING_LEARNING_RATE,
            iterations=settings.RANKING_ITERATIONS,
            depth=10,
            scale_pos_weight=settings.RANKING_SCALE_POS_WEIGHT,
            early_stopping_rounds=settings.RANKING_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )


class RankingModelTrainer:
    def __init__(self, model, train_dataset, eval_dataset) -> None:
        self._model = model

        self._X_train, self._y_train = train_dataset
        self._X_val, self._y_val = eval_dataset

        self._train_dataset, self._eval_dataset = self._initialize_dataset(
            train_dataset, eval_dataset
        )

    def get_model(self):
        return self._model

    def _initialize_dataset(self, train_dataset, eval_dataset):
        X_train, y_train = train_dataset
        X_val, y_val = eval_dataset

        cat_features = list(X_train.select_dtypes(include=["string", "object"]).columns)

        pool_train = Pool(X_train, y_train, cat_features=cat_features)
        pool_val = Pool(X_val, y_val, cat_features=cat_features)

        return pool_train, pool_val

    def fit(self):
        self._model.fit(
            self._train_dataset,
            eval_set=self._eval_dataset,
        )

        return self._model

    def evaluate(self, log: bool = False):
        preds = self._model.predict(self._eval_dataset)

        precision, recall, fscore, _ = precision_recall_fscore_support(
            self._y_val, preds, average="binary"
        )

        if log:
            logger.info(classification_report(self._y_val, preds))

        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    def get_feature_importance(self) -> dict:
        feat_to_score = {
            feature: score
            for feature, score in zip(
                self._X_train.columns,
                self._model.feature_importances_,
            )
        }

        feat_to_score = dict(
            sorted(
                feat_to_score.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return feat_to_score
