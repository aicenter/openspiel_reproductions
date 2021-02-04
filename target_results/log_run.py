''' add an un-logged local file to wandb '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import wandb, csv

FLAGS = flags.FLAGS

flags.DEFINE_string("csv", "", "csv file to log to wandb")
flags.DEFINE_string("project", "openspiel", "project name")

def main(argv):
    wandb.init(project=FLAGS.project)
    wandb.config.update(flags.FLAGS)

    with open(FLAGS.csv, "r") as f:
        cr = csv.reader(f)
        i, unit = next(cr)
        for row in cr:
            wandb.log({i : row[0], unit : row[1]})
            logging.info("{}: {} {}: {}".format(i, row[0], unit, row[1]))

if __name__ == "__main__":
    app.run(main)
        