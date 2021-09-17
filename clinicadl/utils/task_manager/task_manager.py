from abc import abstractmethod

import pandas as pd
import torch

from clinicadl.utils.metric_module import MetricModule


# TODO: add function to check that the output size of the network corresponds to what is expected to
#  perform the task
class TaskManager:
    def __init__(self, mode, n_classes=None):
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
    def generate_test_row(self, idx, data, outputs):
        """
        Computes an individual row of the prediction TSV file.

        Args:
            idx (int): index of the individual input and output in the batch.
            data (dict): input batch generated by a DataLoader on a CapsDataset.
            outputs (torch.Tensor): output batch generated by a forward pass in the model.
        Returns:
            row (List[List[object]]): list of items to be contained in a row of the
                prediction TSV file.
        """
        pass

    @abstractmethod
    def compute_metrics(self, results_df):
        """
        Compute the metrics based on the result of generate_test_row

        Args:
            results_df (pd.DataFrame): results generated based on _results_test_row
        Returns:
            Dict[str, float]
        """
        pass

    @abstractmethod
    def ensemble_prediction(
        self,
        performance_df,
        validation_df,
        selection_threshold=None,
        use_labels=True,
        method="soft",
    ):
        """
        Compute the results at the image-level by assembling the results on parts of the image.

        Args:
            performance_df (pd.DataFrame): results that need to be assembled.
            validation_df (pd.DataFrame): results on the validation set used to compute the performance
                of each separate part of the image.
            selection_threshold (float): with soft-voting method, allows to exclude some parts of the image
                if their associated performance is too low.
            use_labels (bool): If True, metrics are computed and the label column values must be different
                from None.
            method (str): method to assemble the results. Current implementation proposes soft or hard-voting.

        Returns:
            df_final (pd.DataFrame) the results on the image level
            results (Dict[str, float]) the metrics on the image level
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_label_code(df, label):
        """
        Generates a label code that links the output node number to label value.

        Args:
            df (pd.DataFrame): meta-data of the training set.
            label (str): name of the column containing the labels.
        Returns:
            (Dict[str, int]) label_code
        """
        pass

    @staticmethod
    @abstractmethod
    def output_size(input_size, df, label):
        """
        Computes the output_size needed to perform the task.

        Args:
            input_size (Sequence[int]): size of the input.
            df (pd.DataFrame): meta-data of the training set.
            label (str): name of the column containing the labels.
        Returns:
            (Sequence[int]) output_size
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_sampler(dataset, sampler_option="random", n_bins=5):
        """
        Returns sampler according to the wanted options.

        Args:
            dataset: (MRIDataset) the dataset to sample from.
            sampler_option: (str) choice of sampler.
            n_bins: (int) number of bins to used for a continuous variable (regression task).
        Returns:
             sampler (torch.utils.data.Sampler): callable given to the training data loader.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_criterion():
        """
        Gives the optimization criterion

        # TODO: implement a choice
        """
        pass

    @staticmethod
    @abstractmethod
    def get_default_network():
        """Returns the default network to use when no architecture is specified."""
        pass

    def test(self, model, dataloader, criterion, use_labels=True):
        """
        Computes the predictions and evaluation metrics.

        Args:
            model (clinicadl.utils.network.network.Network): the model trained.
            dataloader (torch.utils.data.DataLoader): wrapper of a dataset.
            criterion (loss): function to calculate the loss.
            use_labels (bool): If True the true_label will be written in output DataFrame
                and metrics dict will be created.
        Returns:
            results_df (pd.DataFrame) results of each input.
            metrics_dict (dict) ensemble of metrics + total loss on mode level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=self.columns)
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss = model.compute_outputs_and_loss(
                    data, criterion, use_labels=use_labels
                )
                total_loss += loss.item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs)
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self.compute_metrics(results_df)
            metrics_dict["loss"] = total_loss
        torch.cuda.empty_cache()

        return results_df, metrics_dict