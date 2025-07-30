import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from placefm.config import get_args
from placefm.dataset.loader import get_dataset
from placefm.utils import seed_everything

from methods.hgi import HGI
from methods.placefm import PlaceFM

if __name__ == '__main__':
    args = get_args()

    poi_graph = get_dataset(args.dataset, args, args.load_path)
    seed_everything(args.seed)

    if args.method == 'placefm':
        agent = PlaceFM(data=poi_graph, args=args)
    elif args.method == 'hgi':
        agent = HGI(data=poi_graph, args=args)

    region_embs = agent.generate_embeddings(verbose=args.verbose)
    # evaluator = Evaluator(args)
    # res_mean, res_std = evaluator.evaluate(region_embs, model_type=args.final_eval_model)
    # args.logger.info(f'Test Mean Accuracy: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')
