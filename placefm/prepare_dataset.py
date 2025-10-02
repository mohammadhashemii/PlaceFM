import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from placefm.config import get_args
from placefm.dataset.loader import get_dataset

if __name__ == '__main__':
    args = get_args()

    poi_graph = get_dataset(args.dataset, args, args.load_path)
