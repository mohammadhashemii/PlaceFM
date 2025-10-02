import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.neural_network import MLPRegressor as mlp
from sklearn.tree import DecisionTreeRegressor as dt
from xgboost import XGBRegressor as xgb

from placefm.utils import seed_everything, plot_absolute_error


class Evaluator:
    """
    A class to evaluate different models on geospatial downstream tasks.

    Params
    ------
    args: Namespace
        Arguments containing configurations for evaluation.
    
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the parameters of the evaluator.
        """
        pass


    def fit_with_val(self, data, dt_model, verbose=False, eval_idx=None):
        """
        Trains a specified machine learning model on provided training data and evaluates its performance on test data.
        Parameters
        ----------
        data : tuple
            A tuple containing (x_train, y_train, x_test, y_test), where x_train and x_test are feature matrices,
            and y_train and y_test are target vectors for training and testing, respectively.
        dt_model : str
            The type of model to train. Supported values are 'rf' (Random Forest), 'xgb' (XGBoost), and 'mlp' (Multi-layer Perceptron).
        verbose : bool, optional
            If True, enables verbose output during training. Default is False.
        eval_idx : int or None, optional
            Random seed for model initialization. Default is None.
        Returns
        -------
        metrics_dict : dict
            A dictionary containing evaluation metrics on the test set:
                - 'mae': Mean Absolute Error
                - 'rmse': Root Mean Squared Error
                - 'r2': R-squared score
        Raises
        ------
        ValueError
            If an unsupported model type is provided.
        """
        
        

        x_train, y_train, train_region_ids, x_test, y_test, test_region_ids = data

        if dt_model not in ['rf', 'xgb', 'mlp', 'dt']:
            raise ValueError(f"Unsupported model type: {dt_model}. Supported types are 'rf', 'xgb', 'mlp', 'dt'.")
        if dt_model == 'rf':
            model = rf(n_estimators=100, random_state=eval_idx)
        elif dt_model == 'xgb':
            model = xgb(n_estimators=100, random_state=eval_idx, verbosity=1 if verbose else 0)
        elif dt_model == 'mlp':
            model = mlp(hidden_layer_sizes=(32, 16), max_iter=200, random_state=eval_idx, verbose=verbose)
        elif dt_model == 'dt':
            model = dt(random_state=eval_idx)

        # model = eval(dt_model)(random_state=eval_idx)
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_abs_error = np.abs(y_train - y_train_pred)
        test_abs_error = np.abs(y_test - y_test_pred)

        abs_error_dict = {region_id: error for region_id, error in zip(train_region_ids, train_abs_error)}
        abs_error_dict.update({region_id: error for region_id, error in zip(test_region_ids, test_abs_error)})

        metrics_dict = {
            'mae': metrics.mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'r2': metrics.r2_score(y_test, y_test_pred)
        }

        return metrics_dict, abs_error_dict

    def load_downstream_task_data(self, embs, path, region_ids, task='pd'):
        """
        Load the downstream task data.

        Params
        ------
        embs: dict
            The embeddings dictionary containing region IDs.
        path: str
            Path to the dataset.
        task: str  
            The downstream task to load data for, e.g., 'pd' for population density.    

        region_ids: np.ndarray
            Array of region IDs to filter the dataset.

        ------  
        """

        df = pd.read_csv(path)

        if task == 'pd':
            # Process population density data
            df = df[['ZCTA', 'Population Density (People per Square Kilometer)']]
            df = df.rename(columns={'Population Density (People per Square Kilometer)': 'target'})
        elif task == 'hp':
            # Process housing price data
            df = df[['RegionName', '2024-08-31']]
            df = df.rename(columns={'RegionName': 'ZCTA', '2024-08-31': 'target'})
        elif task == 'pv':
            # Process property value data
            df = df[['place', '2022']]
            df = df.rename(columns={'place': 'ZCTA'})
            # Extract the numeric part of the ZCTA column using regex
            df['ZCTA'] = df['ZCTA'].astype(str).str.extract(r'(\d{5})')[0]
            df = df.rename(columns={'2022': 'target'})

        df['ZCTA'] = df['ZCTA'].astype(str).str.zfill(5)    
        region_ids_str = [str(z).zfill(5) for z in region_ids]

        mask = np.isin(region_ids_str, df['ZCTA'].values)
        embs = embs[mask]
        region_ids_str = np.array(region_ids_str)[mask]

        df = df[df['ZCTA'].isin(region_ids_str)].reset_index(drop=True)
        targets = df['target'].values
        region_ids = df['ZCTA'].values

        # Split into train and test
        num_samples = len(embs)
        split_indices = np.arange(num_samples)
        np.random.shuffle(split_indices)
        split = int(num_samples * 0.8)
        train_idx, test_idx = split_indices[:split], split_indices[split:]

        x_train = embs[train_idx]
        x_test = embs[test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        train_region_ids = region_ids[train_idx]
        test_region_ids = region_ids[test_idx]

        return x_train, y_train, train_region_ids, x_test, y_test, test_region_ids

    def evaluate(self, embs, task='pd', verbose=False):
        """
        Evaluate the embeddings on the specified model type.

        Params
        ------
        embs: torch.Tensor
            The embeddings to evaluate.
        model_type: str
            The type of model to use for evaluation.
        verbose: bool
            If True, print detailed evaluation information.
        task: str
            The downstream task to evaluate on, e.g., 'pd' for population density prediction.

        Returns
        -------
        tuple
            Mean and standard deviation of the evaluation results.
        """
        
        args = self.args    

        if task not in ['pd', 'pv', 'hp']:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are 'pd', 'pv', 'hp'.")

        args.logger.info(f"Evaluating on downstream task: {task}")

        run_eval = args.run_eval


         # Get region_ids from embs
        region_ids = embs['region_id']  # assuming embs is a dict or has attribute 'region_id'
        region_ids = np.array(region_ids)

        res = {
            'mae': [],
            'rmse': [],
            'r2': []}
        
        if verbose:
                print(f" ======== Testing with model type: {args.dt_model} in {args.run_eval} runs ========")
        for i in range(run_eval):
            
            seed_everything(args.seed + i)

            # Check if embeddings are not a numpy array, then transfer to CPU
            if not isinstance(embs['x'], np.ndarray) and hasattr(embs['x'], 'device'):
                embs['x'] = embs['x'].cpu()
            
            if task == 'hp':
                path = f"{args.dt_load_path}/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
            elif task == 'pd':
                path = f"{args.dt_load_path}/zcta_pd.csv"
            elif task == 'pv':
                path = f"{args.dt_load_path}/zcta_pv.csv"
            x_train, y_train, train_region_ids, x_test, y_test, test_region_ids = self.load_downstream_task_data(embs=embs['x'], 
                                                                            path=path, 
                                                                            region_ids=region_ids, 
                                                                            task=task)
            
            
            data = [x_train, y_train, train_region_ids, x_test, y_test, test_region_ids]
            # train and evaluate the ML model on the downstream task data
            metrics_dict, abs_error_dict= self.fit_with_val(data, dt_model=args.dt_model, verbose=verbose, eval_idx=i)
            for key in metrics_dict:
                res[key].append(metrics_dict[key])  

        # compute mean and std for each metric
        for key in res:
            res[key] = np.array(res[key])
            res[key] = (res[key].mean(axis=0), res[key].std(axis=0))
        
        
        # Log and return mean and std of results
        args.logger.info(f'Test RMSE: {res["rmse"][0]:.4f} +/- {res["rmse"][1]:.4f}'
                 f' | Test MAE: {res["mae"][0]:.4f} +/- {res["mae"][1]:.4f}'
                 f' | Test R2: {res["r2"][0]:.4f} +/- {res["r2"][1]:.4f}'
        )

        return res
