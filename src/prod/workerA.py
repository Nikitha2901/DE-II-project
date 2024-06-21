from celery import Celery
import json
from celery.utils.log import get_task_logger


# Model configuration
model_file = './FINAL_MODEL.pickle'
data_file = './test.json'
# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'
logger = get_task_logger(__name__)


def load_data():
    with open(data_file, 'r') as f:
        data = json.load(f)
    y = [item.get('stargazers_count') for item in data]
    # y = np.asarray(y, dtype=np.)
    return data, y


def load_model():
    import ray

    model = None
    with open(model_file, "rb") as rf:
        model = ray.cloudpickle.load(rf)
    assert model is not None
    return model


# Initialize Celery
celery = Celery(
        'workerA',
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND
        )


@celery.task
def get_predictions():
    import numpy as np

    results = {}
    logger.info('RESULTS init: {0}'.format(results))
    X, y = load_data()
    loaded_model = load_model()
    predictions = np.round(
            loaded_model.predict(X)
            ).flatten().astype(np.double)
    results['y'] = list(y)
    results['predicted'] = list(predictions)

    return results


@celery.task
def get_accuracy():
    from sklearn.metrics import r2_score, mean_absolute_error

    X, y = load_data()
    loaded_model = load_model()

    pred = loaded_model.predict(X)

    score = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    return score, r2
