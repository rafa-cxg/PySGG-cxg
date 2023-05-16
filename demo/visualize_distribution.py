import argparse
import os

from pysgg.utils.imports import import_file

from pysgg.data.transforms import build_transforms

from pysgg.config import paths_catalog
from pysgg.data.build import build_dataset

from pysgg.data import get_dataset_statistics
from pysgg.config.defaults import  _C as cfg
from pysgg.data import datasets as D

def main():

    # torch.multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        default='True',
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'], default='Kmeans')

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    is_train = True

    dataset_name = (cfg.DATASETS.TRAIN)[0]
    data_statistics_name = ''.join(dataset_name) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))
    paths_catalog = import_file(
        "pysgg.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    data = DatasetCatalog.get(dataset_name, cfg)
    factory = getattr(D, data["factory"])
    args = data["args"]
    dataset = factory(**args)

    dataset.get_statistics()
    # statistics = get_dataset_statistics(cfg)
    # DatasetCatalog = paths_catalog.DatasetCatalog
    # transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    # datasets = build_dataset(cfg, dataset, transforms, DatasetCatalog, is_train)
if __name__ == "__main__":
    # os.environ["OMP_NUM_THREADS"] = "12"


    main()