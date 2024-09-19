import argparse
import sys

from torch.utils.data import DataLoader

from utils.util import enumerate_with_estimate
from datasets_classify import LunaDataset
from utils.log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LunaCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=1024,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        self.prep_dl = DataLoader(
            LunaDataset(),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerate_with_estimate(
            self.prep_dl,
            "Stuffing cache",
            start_idx=self.prep_dl.num_workers,
        )

        for _ in batch_iter:
            pass

if __name__ == '__main__':
    LunaCacheApp.main()
