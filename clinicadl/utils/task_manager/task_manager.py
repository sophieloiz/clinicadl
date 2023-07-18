from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Sampler

from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.metric_module import MetricModule
from clinicadl.utils.network.network import Network


# TODO: add function to check that the output size of the network corresponds to what is expected to
#  perform the task
class TaskManager:
    def __init__(self, mode: str, n_classes: int = None):
        self.mode = mode
        self.metrics_module = MetricModule(self.evaluation_metrics, n_classes=n_classes)

    @property
    @abstractmethod
    def columns(self):
        """
        List of the columns' names in the TSV file containing the predictions.
        """
        pass

    @property
    @abstractmethod
    def evaluation_metrics(self):
        """
        Evaluation metrics which can be used to evaluate the task.
        """
        pass

    @property
    @abstractmethod
    def save_outputs(self):
        """
        Boolean value indicating if the output values should be saved as tensor for this task.
        """
        pass

    @abstractmethod
    def generate_test_row(
        self, idx: int, data: Dict[str, Any], outputs: Tensor
    ) -> List[List[Any]]:
        """
        Computes an individual row of the prediction TSV file.

        Args:
            idx: index of the individual input and output in the batch.
            data: input batch generated by a DataLoader on a CapsDataset.
            outputs: output batch generated by a forward pass in the model.
        Returns:
            list of items to be contained in a row of the prediction TSV file.
        """
        pass

    @abstractmethod
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute the metrics based on the result of generate_test_row

        Args:
            results_df: results generated based on _results_test_row
        Returns:
            dictionary of metrics
        """
        pass

    @abstractmethod
    def ensemble_prediction(
        self,
        performance_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        selection_threshold: float = None,
        use_labels: bool = True,
        method: str = "soft",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute the results at the image-level by assembling the results on parts of the image.

        Args:
            performance_df: results that need to be assembled.
            validation_df: results on the validation set used to compute the performance
                of each separate part of the image.
            selection_threshold: with soft-voting method, allows to exclude some parts of the image
                if their associated performance is too low.
            use_labels: If True, metrics are computed and the label column values must be different
                from None.
            method: method to assemble the results. Current implementation proposes soft or hard-voting.

        Returns:
            the results and metrics on the image level
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_label_code(df: pd.DataFrame, label: str) -> Optional[Dict[str, int]]:
        """
        Generates a label code that links the output node number to label value.

        Args:
            df: meta-data of the training set.
            label: name of the column containing the labels.
        Returns:
            label_code
        """
        pass

    @staticmethod
    @abstractmethod
    def output_size(
        input_size: Sequence[int], df: pd.DataFrame, label: str
    ) -> Sequence[int]:
        """
        Computes the output_size needed to perform the task.

        Args:
            input_size: size of the input.
            df: meta-data of the training set.
            label: name of the column containing the labels.
        Returns:
            output_size
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_sampler(
        dataset: CapsDataset, sampler_option: str = "random", n_bins: int = 5
    ) -> Sampler:
        """
        Returns sampler according to the wanted options.

        Args:
            dataset: the dataset to sample from.
            sampler_option: choice of sampler.
            n_bins: number of bins to used for a continuous variable (regression task).
        Returns:
             callable given to the training data loader.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_criterion(criterion: str = None) -> _Loss:
        """
        Gives the optimization criterion.
        Must check that it is compatible with the task.

        Args:
            criterion: name of the loss as written in Pytorch.
        Raises:
            ClinicaDLArgumentError: if the criterion is not compatible with the task.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_default_network() -> Network:
        """Returns the default network to use when no architecture is specified."""
        pass

    def test(
        self,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        use_labels: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Args:
            model: the model trained.
            dataloader: wrapper of a CapsDataset.
            criterion: function to calculate the loss.
            use_labels: If True the true_label will be written in output DataFrame
                and metrics dict will be created.
        Returns:
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=self.columns)
        total_loss = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # initialize the loss list to save the loss components
                if i == 0:
                    outputs, loss_dict = model.compute_outputs_and_loss(
                        data, criterion, use_labels=use_labels
                    )
                    for loss_component in loss_dict.keys():
                        total_loss[loss_component] = 0
                    for loss_component in total_loss.keys():
                        total_loss[loss_component] += loss_dict[loss_component].item()
                else:
                    outputs, loss_dict = model.compute_outputs_and_loss(
                        data, criterion, use_labels=use_labels
                    )
                    for loss_component in total_loss.keys():
                        total_loss[loss_component] += loss_dict[loss_component].item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs)
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self.compute_metrics(results_df)
            for loss_component in total_loss.keys():
                metrics_dict[loss_component] = total_loss[loss_component]
        torch.cuda.empty_cache()

        return results_df, metrics_dict
    
    def test_mt(
            self,
            model: Network,
            dataloader: DataLoader,
            criterion: _Loss,
            use_labels: bool = True,
        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
            """
            Computes the predictions and evaluation metrics.

            Args:
                model: the model trained.
                dataloader: wrapper of a CapsDataset.
                criterion: function to calculate the loss.
                use_labels: If True the true_label will be written in output DataFrame
                    and metrics dict will be created.
            Returns:
                the results and metrics on the image level.
            """
            model.eval()
            dataloader.dataset.eval()

            results_df = pd.DataFrame(columns=self.columns)
            results_df2 = pd.DataFrame(columns=self.columns)
            results_df3 = pd.DataFrame(columns=self.columns)
            total_loss = {}
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # initialize the loss list to save the loss components
                    if i == 0:
                        outputs, outputs_2, outputs_3, loss_dict = model.compute_outputs_and_loss_multi(
                            data, criterion, use_labels=use_labels
                        )
                        for loss_component in loss_dict.keys():
                            total_loss[loss_component] = 0
                        for loss_component in total_loss.keys():
                            total_loss[loss_component] += loss_dict[loss_component].item()
                    else:
                        outputs, outputs_2, outputs_3, loss_dict = model.compute_outputs_and_loss_multi(
                            data, criterion, use_labels=use_labels
                        )
                        for loss_component in total_loss.keys():
                            total_loss[loss_component] += loss_dict[loss_component].item()

                    # Generate detailed DataFrame
                    for idx in range(len(data["participant_id"])):
                        row_task1 = self.generate_test_row(idx, data, outputs)
                        row_task2 = self.generate_test_row_mt(idx, data, outputs_2)
                        row_task3 = self.generate_test_row_mt2(idx, data, outputs_3)
                        # row = [row_task1[0] + row_task2[0][3:]]
                        row_df = pd.DataFrame(row_task1, columns=self.columns)
                        row_df2 = pd.DataFrame(row_task2, columns=self.columns)
                        row_df3 = pd.DataFrame(row_task3, columns=self.columns)
                        results_df = pd.concat([results_df, row_df])
                        results_df2 = pd.concat([results_df2, row_df2])
                        results_df3 = pd.concat([results_df3, row_df3])

                    del outputs, outputs_2, outputs_3, loss_dict
                results_df.reset_index(inplace=True, drop=True)
                results_df2.reset_index(inplace=True, drop=True)
                results_df3.reset_index(inplace=True, drop=True)


            if not use_labels:
                metrics_dict = None
            else:
                metrics_dict = self.compute_metrics(results_df)
                print(metrics_dict)

                metrics_dict2 = self.compute_metrics(results_df2)
                print(metrics_dict2)

                metrics_dict3 = self.compute_metrics(results_df3)
                print(metrics_dict3)

                for loss_component in total_loss.keys():
                    metrics_dict[loss_component] = total_loss[loss_component]
                    metrics_dict2[loss_component] = total_loss[loss_component]
                    metrics_dict3[loss_component] = total_loss[loss_component]

            torch.cuda.empty_cache()

            return results_df, results_df2, results_df3, metrics_dict, metrics_dict2, metrics_dict3
