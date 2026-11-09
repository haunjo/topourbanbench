# Example outputs

Scenes generated from the Town10 demo map with the commands in the main [README](../README.md#-usage).

- `topo_ft/`: generated with `./scripts/generate_crowdes_trajectories.sh --scene carla_town10_049 --randomize`
- `standard_ft/`: the same map with the queue, cohesive, and gathering archetypes disabled

Each scene has three files: the trajectory CSV, the group graph JSON, and a bird's-eye preview of the last frame with
the expert paths (a JPEG copy of `_last_frame.png`).

`python demo.py --example` draws the input and output of the Topo-FT scene from these files.
