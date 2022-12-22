import click
from typing import Dict, Tuple, List, Union, Optional
from clinicadl.utils import cli_param


@click.command(name="trivial_motion", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.use_uncropped_image
@cli_param.option.acq_label
@cli_param.option.suvr_reference_region
@click.option(
    "--motion_type",
    type=click.Choice(["image", "kspace"]),
    required=True,
    help="Motion type : image or kspace",
)
@click.option(
    "--translation",
    type=float,
    multiple=2,
    default= [2,4],

    help="Range in mm for the translation",
)
@click.option(
    "--rotation",
    type=float,
    multiple=2,
    default= [2,4],
    help="Range in degree for the rotation",
)
@click.option(
    "--num_transforms",
    type=int,
    default=2,
    help="Number of transforms",
)
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    n_subjects,
    use_uncropped_image,
    acq_label,
    suvr_reference_region,
    motion_type,
    translation,
    rotation,
    num_transforms
):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_motion_dataset

    generate_motion_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing=preprocessing,
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        uncropped_image=use_uncropped_image,
        acq_label=acq_label,
        suvr_reference_region=suvr_reference_region,
        motion_type=motion_type,
        translation=translation,
        rotation=rotation,
        num_transforms=num_transforms,
    )


if __name__ == "__main__":
    cli()
