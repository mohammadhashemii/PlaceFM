import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn import metrics

from placefm.utils import seed_everything


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
        
        

        x_train, y_train, x_test, y_test = data

        if dt_model not in ['rf', 'xgb', 'mlp']:
            raise ValueError(f"Unsupported model type: {dt_model}. Supported types are 'rf', 'xgb', 'mlp'.")
        
        model = eval(dt_model)(random_state=eval_idx)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics_dict = {
            'mae': metrics.mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
            'r2': metrics.r2_score(y_test, y_pred)
        }

        return metrics_dict

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
            df = df[['ZCTA', 'Median Value of Owner Occupied Units (Dollars)']]
            df = df.rename(columns={'Median Value of Owner Occupied Units (Dollars)': 'target'})

        df['ZCTA'] = df['ZCTA'].astype(str).str.zfill(5)    
        # Filter the dataframe to include only the specified region_ids
        region_ids_str = [str(z).zfill(5) for z in region_ids]
        df = df[df['ZCTA'].isin(region_ids_str)]

        targets = df['target'].values

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
        

        return x_train, y_train, x_test, y_test

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

        if task not in ['pd', 'uf', 'hp']:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are 'pd', 'uf', 'hp'.")

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


            x_train, y_train, x_test, y_test = self.load_downstream_task_data(embs=embs['x'], 
                                                                            path=args.dt_load_path, 
                                                                            region_ids=region_ids, 
                                                                            task=task)
            
            
            data = [x_train, y_train, x_test, y_test]
            # train and evaluate the ML model on the downstream task data
            metrics_dict = self.fit_with_val(data, dt_model=args.dt_model, verbose=verbose, eval_idx=i)
            for key in metrics_dict:
                res[key].append(metrics_dict[key])  
        


        # compute mean and std for each metric
        for key in res:
            res[key] = np.array(res[key])
            res[key] = (res[key].mean(axis=0), res[key].std(axis=0))
        
        
        # Log and return mean and std of results
        args.logger.info(f'Test RMSE: {res["rmse"][0]:.2f} +/- {res["rmse"][1]:.2f}'
                 f' | Test MAE: {res["mae"][0]:.2f} +/- {res["mae"][1]:.2f}'
                 f' | Test R2: {res["r2"][0]:.2f} +/- {res["r2"][1]:.2f}'
        )

        return res
