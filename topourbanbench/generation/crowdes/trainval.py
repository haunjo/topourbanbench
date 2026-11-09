import sys
import os
import argparse

# Add the crowdes directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.config import get_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/model/CrowdES_zara2.yaml', help='Path to a model config file')
    parser.add_argument('--dataset_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--trainer_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--model_train', type=str, default='emitter', help='Stage of the experiment', choices=['emitter_pre', 'emitter', 'simulator'])
    parser.add_argument('--test', default=False, action='store_true', help='Evaluation mode.')
    parser.add_argument('--export', default=False, action='store_true', help='Visualization mode.')
    parser.add_argument('--synthetic', default=False, action='store_true', help='Use synthetic dataset for inference.')
    # Options for --synthetic; given values override the defaults in CrowdES/evaluate_synthetic_dataset.py
    parser.add_argument('--scene', type=str, default=None, help='Generate only this scene.')
    parser.add_argument('--town', type=str, default=None, help='Generate only the scenes of this town, e.g. town10.')
    parser.add_argument('--dataset_root', type=str, default=None, help='Directory with the extracted maps.')
    parser.add_argument('--out_subset', type=str, default=None, help='Output subset (standard_ft or topo_ft). Detected from configs/scenarios.yaml by default.')
    parser.add_argument('--randomize', default=False, action='store_true', help='Randomize group parameters per scene.')
    parser.add_argument('--force', default=False, action='store_true', help='Regenerate scenes whose CSV already exists.')
    args = parser.parse_args()

    config = get_config(args.model_config, args.dataset_config, args.trainer_config)
    
    # Print the arguments and configs
    print('===== Arguments =====')
    print_arguments(vars(args))

    print('===== Configs =====')
    print_arguments(config)

    # Import the appropriate pipeline
    if args.test:
        from CrowdES.evaluate import *
    elif args.export:
        from CrowdES.evaluate_export_generated_traj import *
    elif args.synthetic:
        from CrowdES.evaluate_synthetic_dataset import *
        import CrowdES.evaluate_synthetic_dataset as synthetic
        if args.scene is not None:
            synthetic.SCENE_LIST = [args.scene]
        if args.town is not None:
            synthetic.TOWN_FILTER = args.town
        if args.dataset_root is not None:
            synthetic.DATASET_ROOT = args.dataset_root
        if args.out_subset is not None:
            synthetic.OUT_SUBSET = args.out_subset
        if args.randomize:
            synthetic.RANDOMIZE_PARAMS = True
        if args.force:
            synthetic.FORCE_REGEN = True
    elif args.model_train == 'emitter_pre':
        from CrowdES.emitter.emitter_pre_trainer import *
    elif args.model_train == 'emitter':
        from CrowdES.emitter.emitter_trainer import *
    elif args.model_train == 'simulator':
        from CrowdES.simulator.simulator_trainer import *
        
    main(config)
