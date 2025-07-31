import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from placefm.config import get_args
from placefm.dataset.loader import get_dataset
from placefm.utils import seed_everything

from methods.hgi import HGI
# from methods.placefm import PlaceFM
from placefm.evaluator.eval_agent import Evaluator

if __name__ == '__main__':
    args = get_args()

    poi_graph = get_dataset(args.dataset, args, args.load_path)
    seed_everything(args.seed)

    if args.method == 'placefm':
        agent = PlaceFM(data=poi_graph, args=args)
    elif args.method == 'hgi':
        agent = HGI(data=poi_graph, args=args)

    region_embs = agent.generate_embeddings(verbose=args.verbose)

    if args.eval:
        evaluator = Evaluator(args)

        tasks = ['pd', 'hp']
        for t in tasks:
            res_dict = evaluator.evaluate(region_embs, task=t, verbose=args.verbose)
