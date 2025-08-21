'''Configuration'''
import os
import sys
import json
import logging
import click
from pprint import pformat


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        # Use pprint's pformat to print the dictionary in a pretty manner
        return pformat(self.__dict__, compact=True)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)


def update_from_dict(obj, updates):
    for key, value in updates.items():
        # set higher priority from command line as we explore some factors
        if key in ['init'] and obj.init is not None:
            continue
        setattr(obj, key, value)



# recommend hyperparameters here
def method_config(args):
    try:
        # print(os.path.dirname(graphslim.__file__))
        conf_dt = json.load(
            open(f"{os.path.join('./configs', args.method, args.dataset)}.json"))
        update_from_dict(args, conf_dt)
    except:
        print('No config file found or error in json format, please use method_config(args)')
    if args.method in ['msgc']:
        args.batch_adj = 16
        # add temporary changes here
        # do not modify the config json

    return args


@click.command()
@click.option('--method', '-M', default='placefm', show_default=True)
@click.option('--gpu_id', '-G', default=0, help='gpu id start from 0, -1 means cpu', show_default=True)
@click.option('--hidden', '-H', default=256, show_default=True)
@click.option('--condense_model', default='SGC',
              type=click.Choice(
                  ['GCN', 'GAT', 'SGC', 'APPNP', 'Cheby', 'GraphSage', 'GAT']
              ), show_default=True)
@click.option('--epochs', '-E', default=100, show_default=True, help='number of training epochs')
@click.option('--lr', default=0.01, show_default=True)
@click.option('--weight_decay', '--wd', default=0.0, show_default=True)
@click.option('--seed', '-S', default=1, help='Random seed', show_default=True)
@click.option('--verbose', '-V', is_flag=True, show_default=True)
@click.option('--init', default='random', help='features initialization methods',
              type=click.Choice(
                  ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC',
                   'affinity_GS', 'kron', 'vng', 'clustering', 'averaging',
                   'cent_d', 'cent_p', 'kcenter', 'herding', 'random']
              ), show_default=True)
@click.option('--optim', default="Adam", show_default=True)
@click.option('--dropout', default=0.0, show_default=True)
@click.option('--ntrans', default=1, show_default=True, help='number of transformations in SGC and APPNP')
@click.option('--with_bn', is_flag=True, show_default=True)
@click.option('--alpha', default=0.1, help='for appnp', show_default=True)
@click.option('--save_path', '--sp', default='../checkpoints', show_default=True, help='save path for synthetic graph')

# ====== evaluation args ======
@click.option('--eval', '-E', is_flag=True, show_default=True, help='whether to evaluate the model after training')
@click.option('--run_eval', '-R', default=10, show_default=True, help='number of runs for evaluation')
@click.option('--dt_model', default='rf', type=click.Choice(['rf', 'xgb', 'mlp']), show_default=True,
              help='the downstream task model to use for final evaluation')
@click.option('--embeddings', '-EM' , help='path to the region embeddings file')

# ====== dataset args ======
@click.option('--dataset', '-D', default='f-osm', show_default=True)
@click.option('--city', '-C', default='atlanta', show_default=True)
@click.option('--edge_creation', default='dt', type=click.Choice(['knn', 'dt']), show_default=True)
@click.option('--split', default='fixed', show_default=True,
              help='only support public split now, do not change it')  # 'fixed', 'random', 'few'
@click.option('--load_path', '--lp', default='../data', show_default=True, help='save path for trained embeddings')
@click.option('--dt_load_path', default='../data/downstream_tasks/zcta_dt.csv', show_default=True, help='downstream task data path')
# ====== PlaceFM args ======
@click.option('--clustering_method', default='kmeans', type=click.Choice(['kmeans', 'dbscan']), show_default=True)
@click.option('--region_agg_method', default='mean', type=click.Choice(['mean', 'max']), show_default=True)
@click.option("--placefm_agg_alpha", default=0.3, show_default=True)
@click.option("--placefm_agg_beta", default=0.9, show_default=True)
@click.option("--placefm_agg_gamma", default=0.0, show_default=True)
@click.option("--placefm_fuzziness", default=1.0, show_default=True)
@click.option("--placefm_rep_fuzz", default=20, show_default=True)
@click.option("--placefm_kmeans_reduction_ratio", default=0.1, show_default=True)

# ====== HGI args ======
@click.option("--attention_head", type=int, default=4, show_default=True)
@click.option('--hgi_alpha', default=0.5, help='the hyperparameter to balance mutual information', show_default=True)
@click.option("--max_norm", type=float, default=0.9, show_default=True)
@click.option('--hgi_gamma', type=float, default=0.9, show_default=True)
@click.option("--warmup_period", type=int, default=40, show_default=True)

@click.pass_context
def cli(ctx, **kwargs):
    args = dict2obj(kwargs)
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
        args.device = f'cuda:0'
    else:
        # if gpu_id=-1, use cpu
        args.device = 'cpu'
    args.save_path = os.path.join(args.save_path)
    path = args.save_path
    # for benchmark, we need unified settings and reduce flexibility of args
    args = method_config(args)
    # setting_config has higher priority than methods_config
    
    for key, value in ctx.params.items():
        if ctx.get_parameter_source(key) == click.core.ParameterSource.COMMANDLINE:
            setattr(args, key, value)
    if not os.path.exists(f'{path}/logs/{args.method}'):
        try:
            os.makedirs(f'{path}/logs/{args.method}')
        except:
            print(f'{path}/logs/{args.method} exists!')
    logging.basicConfig(filename=f'{path}/logs/{args.method}/{args.dataset}.log',
                        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args.logger = logging.getLogger(__name__)
    args.logger.addHandler(logging.StreamHandler())
    args.logger.info(args)
    return args


def get_args():
    return cli(standalone_mode=False)


if __name__ == '__main__':
    cli()
