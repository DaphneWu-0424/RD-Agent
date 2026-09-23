from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from rdagent.core.utils import cache_with_pickle

pandarallel.initialize(verbose=1)

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int,
    ) -> pd.DataFrame:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1,
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1]),
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    @staticmethod
    def _smoke_date_env() -> dict[str, str]:
        """Build bounded train/valid/test segments from the Profile dataset."""
        profile_path = Path(FACTOR_COSTEER_SETTINGS.data_folder_profile) / "daily_pv.h5"
        if not profile_path.is_file():
            raise FactorEmptyError(f"Smoke profile data does not exist: {profile_path}")
        profile_index = pd.read_hdf(profile_path, key="data").index
        dates = pd.DatetimeIndex(profile_index.get_level_values("datetime").unique()).sort_values()
        if len(dates) < 10:
            raise FactorEmptyError("Smoke profile data needs at least 10 distinct trading dates.")
        train_end_idx = max(0, int(len(dates) * 0.6) - 1)
        valid_start_idx = train_end_idx + 1
        valid_end_idx = max(valid_start_idx, int(len(dates) * 0.8) - 1)
        test_start_idx = valid_end_idx + 1
        return {
            "train_start": dates[0].strftime("%Y-%m-%d"),
            "train_end": dates[train_end_idx].strftime("%Y-%m-%d"),
            "valid_start": dates[valid_start_idx].strftime("%Y-%m-%d"),
            "valid_end": dates[valid_end_idx].strftime("%Y-%m-%d"),
            "test_start": dates[test_start_idx].strftime("%Y-%m-%d"),
            "test_end": dates[-1].strftime("%Y-%m-%d"),
        }

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """
        research_mode = getattr(exp, "research_mode", "full")
        factor_data_type = "Profile" if research_mode == "smoke" else "All"
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].research_mode = research_mode
            logger.info("Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        fbps = FactorBasePropSetting()
        env_to_use = {
            "PYTHONPATH": "./",
            "train_start": fbps.train_start,
            "train_end": fbps.train_end,
            "valid_start": fbps.valid_start,
            "valid_end": fbps.valid_end,
            "test_start": fbps.test_start,
            "feature_names": str(list(exp.base_features.keys())),
            "feature_expressions": str(list(exp.base_features.values())),
        }
        if fbps.test_end is not None:
            env_to_use.update({"test_end": fbps.test_end})
        if research_mode == "smoke":
            smoke_dates = self._smoke_date_env()
            env_to_use.update(smoke_dates)
            logger.warning(
                f"Performance gate restricted this experiment to smoke research: {smoke_dates}",
            )

        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info("SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list, data_type=factor_data_type)

            # Process the new factors data
            logger.info("New factor processing ...")
            new_factors = process_factor_data(exp, data_type=factor_data_type)

            if new_factors.empty:
                raise FactorEmptyError(f"Factors failed to run on the {research_mode} sample.")
            exp.injected_factor_columns = [str(column) for column in new_factors.columns]
            logger.info(
                f"[FactorRunner] injected {research_mode} factor(s): "
                f"{', '.join(exp.injected_factor_columns)}",
            )

            # Combine the SOTA factor and new factors if SOTA factor exists
            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors.",
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            # Sort and nest the combined factors under 'feature'
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            logger.info("Factor data processing completed.")

            num_features = len(exp.base_features) + len(combined_factors.columns)

            # Due to the rdagent and qlib docker image in the numpy version of the difference,
            # the `combined_factors_df.pkl` file could not be loaded correctly in qlib dokcer,
            # so we changed the file type of `combined_factors_df` from pkl to parquet.
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # Save the combined factors to the workspace
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # If model exp exists in the previous experiment
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info("Experiment execution ...")
            if exist_sota_model_exp:
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]},
                )
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                if sota_training_hyperparameters:
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        },
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                if sota_model_type == "TimeSeries":
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20},
                    )
                elif sota_model_type == "Tabular":
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # model + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use,
                )
            else:
                # LGBM + combined factors
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors.yaml",
                    run_env=env_to_use,
                )
        else:
            logger.info("Experiment execution ...")
            if exp.base_feature_codes:
                factors = process_factor_data(exp, data_type=factor_data_type)
                factors = factors.sort_index()
                factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], factors.columns])
                factors.columns = new_columns
                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
                # Save the combined factors to the workspace
                factors.to_parquet(target_path, engine="pyarrow")
                logger.info("Factor data processing completed.")
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors.yaml",
                    run_env=env_to_use,
                )
            else:
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline.yaml",
                    run_env=env_to_use,
                )

        if result is None:
            logger.error(f"Failed to run this experiment, because {stdout}")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        exp.result = result
        exp.research_mode = research_mode
        exp.research_window = {
            key: env_to_use[key]
            for key in ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
        }
        exp.stdout = stdout

        return exp
