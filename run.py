import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="experiment.yaml", help="Experiment config file")
parser.add_argument("--jobdir", default="jobs", help="Where to store generated slurm jobfiles and logs")
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.cfg, "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["run"]:
        exp_config = config[experiment]
        jobfile = os.path.join(args.jobdir, "{}.job".format(experiment))
        jobargs = (config["container"], exp_config["script"], exp_config["flagfile"])
        cmd = "singularity exec {} python3 {} --flagfile {}".format(*jobargs)
        
        with open(jobfile, "w") as f:
            f.writelines("#!/bin/bash\n")
            f.writelines("#SBATCH --partition {}\n".format(exp_config["partition"]))
            f.writelines("#SBATCH --nodes {}\n".format(exp_config["nodes"]))
            f.writelines("#SBATCH --ntasks-per-node {}\n".format(exp_config["ntasks-per-node"]))
            f.writelines("#SBATCH --mem-per-cpu {}\n".format(exp_config["mem-per-cpu"]))
            f.writelines("#SBATCH --time {}\n".format(exp_config["time"]))
            f.writelines("#SBATCH --job-name {}\n".format(exp_config["job-name"]))
            f.writelines("#SBATCH --output {}\n".format(exp_config["output"]))
            f.writelines("module purge\n")
            f.writelines(cmd)
        
        os.system("sbatch {}".format(jobfile))