import os
import sys
import torch

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from placefm.config import get_args
from placefm.utils import seed_everything
from placefm.evaluator.eval_agent import Evaluator

if __name__ == '__main__':
    args = get_args()

    seed_everything(args.seed)

    # load the region embeddings
    region_embs = torch.load(args.embeddings)

    evaluator = Evaluator(args)

    tasks = ['pd', 'hp']
    for t in tasks:
        res_dict = evaluator.evaluate(region_embs, task=t, verbose=args.verbose)
