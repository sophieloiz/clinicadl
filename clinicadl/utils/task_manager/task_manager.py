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
        import numpy as np

        model.eval()
        # dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=self.columns)
        total_loss = 0
        embedded_features = None
        print(dataloader)
        import matplotlib.pyplot as plt

        features_list = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("Remove alpha from task manager if no ssda training")
                outputs, loss_dict, features = model.compute_outputs_and_loss(
                    data, criterion, use_labels=use_labels
                )  # , alpha=0

                features_np = features.cpu().numpy()
                print(features_np.shape)
                features_flat = features_np.reshape(
                    features_np.shape[0], -1
                )  # Flatten the features
                # features_flat = features_np.flatten()
                features_list.append(features_flat)

                # )
                # import frequency_feature_map_visualization as fv
                # feature_map_dict = fv.visualize_feature_maps_3d(model, data["image"], device=torch.device('cpu'))
                # print(feature_map_dict)
                # fv.save_feature_maps_to_npy(feature_map_dict, f'/export/home/cse180022/apprimage_sophie/Distangle_Guanghui/saved_feature_maps2/')

                total_loss += loss_dict["loss"].item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs)
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
            results_df.reset_index(inplace=True, drop=True)

        plt.figure()
        from sklearn.manifold import TSNE

        features_flat_all = features_list.reshape(
            features_list.shape[0], -1
        )  # Flatten the f
        tsne = TSNE(n_components=2, random_state=42)
        embedded_batch = tsne.fit_transform(features_flat_all)
        print(embedded_batch)

        plt.scatter(embedded_batch[:, 0], embedded_batch[:, 1])
        plt.title("t-SNE Visualization")
        plt.savefig("/export/home/cse180022/test_tsne.pdf", dpi=150)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self.compute_metrics(results_df)
            metrics_dict["loss"] = total_loss
        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def test_da(
        self,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        alpha: float,
        target: bool = False,
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
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss_dict = model.compute_outputs_and_loss_test(
                    data, criterion, alpha, target
                )
                total_loss += loss_dict["loss"].item()

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
            metrics_dict["loss"] = total_loss
        torch.cuda.empty_cache()

        return results_df, metrics_dict
