from celery import Celery
from misc.viz_ploration.backend.algorithm import approx

app = Celery(__name__, backend='rpc://', broker='redis://localhost:6379')

@app.task
def integrate(*args, **kwargs):
    try:
        return approx(*args, **kwargs)
    except Exception as e:
        print(e)
        return