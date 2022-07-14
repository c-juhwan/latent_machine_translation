import pandas as pd
from nlgeval import NLGEval

def post_processing(args):

    nlgeval = NLGEval()  # loads the models
    metrics_dict = nlgeval.compute_metrics(references, hypothesis)