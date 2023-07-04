from logging import getLogger

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import sampler

from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl.task_manager")

from clinicadl.utils.task_manager.task_manager import TaskManager


class ClassificationManager(TaskManager):
    def __init__(
        self,
        mode,
        n_classes=None,
        df=None,
        label=None,
        label2=None
    ):
        if n_classes is None:
            n_classes = self.output_size(None, df, label)
        self.n_classes = n_classes

        super().__init__(mode, n_classes)

    @property
    def columns(self):
        return [
            "participant_id",
            "session_id",
            f"{self.mode}_id",
            "true_label",
            "predicted_label",
        ] + [f"proba{i}" for i in range(self.n_classes)]

    @property
    def columns_mt(self):
        return [
            "participant_id",
            "session_id",
            f"{self.mode}_id",
            "true_label",
            "predicted_label",
        ] + [f"proba{i}" for i in range(self.n_classes)] +[ "true_label2",
            "predicted_label2",
        ] + [f"proba{i}2" for i in range(self.n_classes)]

    @property
    def evaluation_metrics(self):
        return ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "BA"]
    
    @property
    def evaluation_metrics_mt(self):
        #return ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "BA","accuracy2", "sensitivity2", "specificity2", "PPV2", "NPV2", "BA2"]
        return ["accuracy", "accuracy2", "sensitivity", "sensitivity2", "specificity", "specificity2","PPV", "PPV2", "NPV", "NPV2", "BA", "BA2"]


    @property
    def save_outputs(self):
        return False

    def generate_test_row(self, idx, data, outputs):
        prediction = torch.argmax(outputs[idx].data).item()
        normalized_output = softmax(outputs[idx], dim=0)
        return [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{self.mode}_id"][idx].item(),
                data["label"][idx].item(),
                prediction,
            ]
            + [normalized_output[i].item() for i in range(self.n_classes)]
        ]
    
    def generate_test_row_mt(self, idx, data, outputs, outputs2):
        prediction = torch.argmax(outputs[idx].data).item()
        prediction2 = torch.argmax(outputs2[idx].data).item()
        normalized_output = softmax(outputs[idx], dim=0)
        normalized_output2 = softmax(outputs2[idx], dim=0)

        return [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{self.mode}_id"][idx].item(),
                data["label"][idx].item(),
                prediction,
                data["label2"][idx].item(),
                prediction2,
            ]
            + [normalized_output[i].item() for i in range(self.n_classes)]
            + [normalized_output2[i].item() for i in range(self.n_classes)]

        ]

    def compute_metrics(self, results_df):
        return self.metrics_module.apply(
            results_df.true_label.values,
            results_df.predicted_label.values,
        )

    def compute_metrics_mt(self, results_df):
        return self.metrics_module.apply(
            results_df.true_label2.values,
            results_df.predicted_label2.values,
        )
    
    # def compute_metrics_mt(self, results_df):
    #     res_1 = self.metrics_module.apply(
    #         results_df.true_label.values,
    #         results_df.predicted_label.values,
    #     )
    #     res_2 = self.metrics_module.apply(
    #         results_df.true_label2.values,
    #         results_df.predicted_label2.values,
    #     )
    #     print(res_1)
    #     print(res_2)

    #     return {**res_1, **res_2}

    @staticmethod
    def generate_label_code(df, label):
        unique_labels = list(set(getattr(df, label)))
        unique_labels.sort()
        return {str(key): value for value, key in enumerate(unique_labels)}

    @staticmethod
    def output_size(input_size, df, label):
        label_code = ClassificationManager.generate_label_code(df, label)
        return len(label_code)

    @staticmethod
    def generate_sampler(dataset, sampler_option="random", n_bins=5):
        df = dataset.df
        labels = df[dataset.label].unique()
        codes = set()
        for label in labels:
            codes.add(dataset.label_code[label])
        count = np.zeros(len(codes))

        for idx in df.index:
            label = df.loc[idx, dataset.label]
            key = dataset.label_fn(label)
            count[key] += 1

        weight_per_class = 1 / np.array(count)
        weights = []

        for idx, label in enumerate(df[dataset.label].values):
            key = dataset.label_fn(label)
            weights += [weight_per_class[key]] * dataset.elem_per_image

        if sampler_option == "random":
            return sampler.RandomSampler(weights)
        elif sampler_option == "weighted":
            return sampler.WeightedRandomSampler(weights, len(weights))
        else:
            raise NotImplementedError(
                f"The option {sampler_option} for sampler on classification task is not implemented"
            )

    def ensemble_prediction(
        self,
        performance_df,
        validation_df,
        selection_threshold=None,
        use_labels=True,
        method="soft",
    ):
        """
        Computes hard or soft voting based on the probabilities in performance_df. Weights are computed based
        on the balanced accuracies of validation_df.

        ref: S. Raschka. Python Machine Learning., 2015

        Args:
            performance_df (pd.DataFrame): Results that need to be assembled.
            validation_df (pd.DataFrame): Results on the validation set used to compute the performance
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

        def check_prediction(row):
            if row["true_label"] == row["predicted_label"]:
                return 1
            else:
                return 0

        if method == "soft":
            # Compute the sub-level accuracies on the validation set:
            validation_df["accurate_prediction"] = validation_df.apply(
                lambda x: check_prediction(x), axis=1
            )
            sub_level_accuracies = validation_df.groupby(f"{self.mode}_id")[
                "accurate_prediction"
            ].mean()
            if selection_threshold is not None:
                sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
            weight_series = sub_level_accuracies / sub_level_accuracies.sum()
        elif method == "hard":
            n_modes = validation_df[f"{self.mode}_id"].nunique()
            weight_series = pd.DataFrame(np.ones((n_modes, 1)))
        else:
            raise NotImplementedError(
                f"Ensemble method {method} was not implemented. "
                f"Please choose in ['hard', 'soft']."
            )

        # Sort to allow weighted average computation
        performance_df.sort_values(
            ["participant_id", "session_id", f"{self.mode}_id"], inplace=True
        )
        weight_series.sort_index(inplace=True)

        # Soft majority vote
        df_final = pd.DataFrame(columns=self.columns)
        for (subject, session), subject_df in performance_df.groupby(
            ["participant_id", "session_id"]
        ):
            label = subject_df["true_label"].unique().item()
            proba_list = [
                np.average(subject_df[f"proba{i}"], weights=weight_series)
                for i in range(self.n_classes)
            ]
            prediction = proba_list.index(max(proba_list))
            row = [[subject, session, 0, label, prediction] + proba_list]
            row_df = pd.DataFrame(row, columns=self.columns)
            df_final = pd.concat([df_final, row_df])

        if use_labels:
            results = self.compute_metrics(df_final)
        else:
            results = None

        return df_final, results

    @staticmethod
    def get_criterion(criterion=None):
        compatible_losses = ["CrossEntropyLoss", "MultiMarginLoss"]
        if criterion is None:
            return nn.CrossEntropyLoss()
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Classification loss must be chosen in {compatible_losses}."
            )
        return getattr(nn, criterion)()

    @staticmethod
    def get_default_network():
        return "Conv5_FC3"
