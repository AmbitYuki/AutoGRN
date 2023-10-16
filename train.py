from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from model.pytorch.supervisor import GTSSupervisor
from lib.utils import load_graph_data

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)
        save_adj_name = args.config_filename[11:-5]
        # print(args.config_filename)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj = load_graph_data(graph_pkl_filename)
        supervisor = GTSSupervisor(adj, temperature=args.temperature, **supervisor_config)
        supervisor.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/para_pjm_10.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
    args = parser.parse_args()
    main(args)

