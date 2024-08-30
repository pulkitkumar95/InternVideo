import os
from datetime import datetime
import argparse
import time
import pandas as pd
import socket
import subprocess

def get_gpu_info(gpu_type=['a6000', 'a5000'], remove_nodes=None, qos='vulc_scav'):
    if qos == 'scav':
        remove_nodes.append('vulcan')
    def run(cmd, print_err=True):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('UTF-8').splitlines()
        except subprocess.CalledProcessError as e:
            # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            if print_err:
                print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            return [cmd.split()[-1]]
    gpudata = run('sinfo -O nodehost,gres -h')
    new_gpu_data = []
    for gpu in gpu_type:
        new_gpu_data += [line.split(' ')[0] for line in gpudata if gpu in line]
    if remove_nodes is not None:
        for node in remove_nodes:
            new_gpu_data = [gpu_node for gpu_node in new_gpu_data if node not in gpu_node]
    assert len(new_gpu_data) > 0, 'No GPU found'
    return ','.join(new_gpu_data)



qos_dict = {"sailon" : {"nhrs" : 2, "cores": 16, "mem":128},
            "scav" : {"nhrs" : 24, "cores": 16, "mem":128},
            "vulc_scav" : {"nhrs" : 24, "cores": 16, "mem":128},
            "cml_scav" : {"nhrs" : 24, "cores": 16, "mem":128}, 

            "high" : {"gpu":4, "cores": 16, "mem":120, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168},
            "tron" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}




def check_qos(args):
    
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--explore', action='store_true')


parser.add_argument('--reverse', action='store_true')
parser.add_argument('--sp_crop', action='store_true')

parser.add_argument('--qos', default="vulc_scav", type=str, help='Qos to run')
parser.add_argument('--env', default="reinforce_cosine_all_boundaries", type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=1, type=int, help='Number of gpus')
parser.add_argument('--cores', default=1, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=20, type=int, help='RAM in G')
parser.add_argument('--gpu_type', default='a4', type=str, help='RAM in G')
parser.add_argument('--tracker_to_use', default='cotracker', type=str, help='RAM in G')










# parser.add_argument('--path', default='/fs/vulcan-projects/actionbytes/vis/ab_training_run3_rerun_32_0.0001_4334_new_dl_nocasl_checkpoint_best_dmap_ab_info.hkl')
# parser.add_argument('--num_ab', default= 100000, type=int, help='number of actionbytes')

args = parser.parse_args()

if 'nexus' in socket.gethostname():
    gpu_types = ['a6000','a5000', 'a4000']

remove_nodes = ['cml17', 'cml20', 'cml28', 'clip', 'gamma', 'vulcan30']

nodes = get_gpu_info(gpu_types,remove_nodes, qos=args.qos)



feat_dump = True

args = parser.parse_args()
args.env += str(int(time.time()))


output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print("Output Directory: %s" % output_dir)
num_splits = 200
params = [(idx) for idx in range(num_splits)]

print(len(params))
pca = True
                        
temporal_skip = None
hostname = socket.gethostname()
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:

    for i, (idx) in enumerate(params):

        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'test_{i}'
     
        cmd = f'python extract_gym.py --num_splits {num_splits} --split_id {idx}'
    
       
        
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')
        #break
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'internvid.slurm')
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{len(params)}\n")
    #slurmfile.write(f"#SBATCH --array=1-10\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    slurmfile.write("#SBATCH --nodes=1\n")
    slurmfile.write("#SBATCH --nodelist=%s\n" % nodes)
    # slurmfile.write("#SBATCH --exclude=vulcan[00-23]\n")

    
    args = check_qos(args)


    if "scav" in args.qos or "tron" in args.qos:
        if args.qos == "scav":
            slurmfile.write("#SBATCH --account=scavenger\n")
            slurmfile.write("#SBATCH --qos scavenger\n")
            slurmfile.write("#SBATCH --partition scavenger\n")


        elif args.qos == "vulc_scav":
            slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
            slurmfile.write("#SBATCH --qos vulcan-scavenger\n")
            slurmfile.write("#SBATCH --partition vulcan-scavenger\n")
        
        elif args.qos == "cml_scav":
            slurmfile.write("#SBATCH --account=cml-scavenger\n")
            slurmfile.write("#SBATCH --qos cml-scavenger\n")
            slurmfile.write("#SBATCH --partition cml-scavenger\n")

            
        
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        if not args.gpu is None:
            # if hostname in {'nexus', 'vulcan'}:
            if args.gpu_type == 'a4':
                gpu_str = 'rtxa4000:'
            elif args.gpu_type == 'a6':
                gpu_str = 'rtxa6000:'
            elif args.gpu_type == 'a5':
                gpu_str = 'rtxa5000:'
            else:
                gpu_str = ''
            slurmfile.write(f'#SBATCH --gres=gpu:{args.gpu}\n')
           
        else:
            raise ValueError("Specify the gpus for scavenger")
    else:
       
        slurmfile.write("#SBATCH --account=abhinav\n")
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --gres=gpu:p6000:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)


    slurmfile.write("#SBATCH --exclude=vulcan[01-08]")
    
    slurmfile.write("\n")
    #slurmfile.write("export MKL_SERVICE_FORCE_INTEL=1\n")p
    slurmfile.write("cd " + os.getcwd() + '\n')
    slurmfile.write("module load ffmpeg\n")
    slurmfile.write("export MKL_THREADING_LAYER=GNU\n")
    if args.tracker_to_use == 'cotracker' or args.tracker_to_use == 'pips2':
        slurmfile.write("source /fs/cfar-projects/actionloc/new_miniconda/bin/activate\n")
        slurmfile.write("conda activate omnivid\n")
    elif args.tracker_to_use == 'tapir':
        slurmfile.write("source /fs/cfar-projects/actionloc/colab/namithap/point_trackers/tapnet/venv_tapir/bin/activate\n")
    # slurmfile.write("cd ./libs/utils\n")
    # slurmfile.write("python setup.py install --user\n")
    # slurmfile.write("cd ../..\n")


    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)
