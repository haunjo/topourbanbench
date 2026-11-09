#!/bin/bash
# Generate CrowdES trajectories for synthetic scenarios
# Usage: ./scripts/generate_crowdes_trajectories.sh [--scene SCENE] [--town TOWN] [--config CONFIG]
#            [--dataset_root DIR] [--out_subset standard_ft|topo_ft] [--randomize] [--force]
#
# Examples:
#   ./scripts/generate_crowdes_trajectories.sh                              # All town10 scenes
#   ./scripts/generate_crowdes_trajectories.sh --scene carla_town10_008    # Single scene
#   ./scripts/generate_crowdes_trajectories.sh --town town03               # Only town03 scenes
#   ./scripts/generate_crowdes_trajectories.sh --town all                  # All towns
#
# Subset toggle (standard_ft vs topo_ft):
#   The output subset is auto-detected from configs/scenarios.yaml:
#     - standard_ft: all crowd scenarios disabled (queue/gathering/cohesive enabled: false)
#     - topo_ft:     any crowd scenario enabled (queue or gathering enabled: true)
#                    Pass --randomize for per-scene group parameter variation.
#   --out_subset overrides the auto-detection.

set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CROWDES_DIR="${PROJECT_ROOT}/topourbanbench/generation/crowdes"

# Default parameters
MODEL_CONFIG="./configs/model/CrowdES_zara2.yaml"
TOWN_FILTER="town10"
SCENE_NAME=""
DATASET_ROOT="./datasets/Synthetic"
OUT_SUBSET=""
RANDOMIZE="false"
FORCE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)        SCENE_NAME="$2";    shift 2 ;;
        --config)       MODEL_CONFIG="$2";  shift 2 ;;
        --town)         TOWN_FILTER="$2";   shift 2 ;;
        --dataset_root) DATASET_ROOT="$2";  shift 2 ;;
        --out_subset)   OUT_SUBSET="$2";    shift 2 ;;
        --randomize)    RANDOMIZE="true";   shift ;;
        --force)        FORCE="true";       shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Convert "all" to empty string (no filter)
if [ "${TOWN_FILTER}" == "all" ]; then
    TOWN_FILTER=""
fi

echo "======================================"
echo "CrowdES Trajectory Generation"
echo "======================================"
echo "Project Root:  ${PROJECT_ROOT}"
echo "CrowdES Dir:   ${CROWDES_DIR}"
echo "Model Config:  ${MODEL_CONFIG}"
if [ -n "${SCENE_NAME}" ]; then
    echo "Scene:         ${SCENE_NAME}"
else
    echo "Town Filter:   ${TOWN_FILTER:-'(all towns)'}"
fi
echo "Dataset Root:  ${DATASET_ROOT}"
echo "Out Subset:    ${OUT_SUBSET:-'(auto)'}"
echo "Randomize:     ${RANDOMIZE}"
echo "Force:         ${FORCE}"
echo "======================================"

# Navigate to crowdes directory
cd "${CROWDES_DIR}"

# Check if checkpoints exist
if [ ! -d "checkpoints" ]; then
    echo ""
    echo "⚠️  WARNING: checkpoints/ directory not found!"
    echo "Please download the official pretrained CrowdES weights first:"
    echo "  https://github.com/InhwanBae/Crowd-Behavior-Generation/releases/tag/v1.0-model"
    echo "  e.g., unzip CrowdES_zara2_pretrained.zip inside topourbanbench/generation/crowdes/"
    echo ""
    exit 1
fi

# Options for trainval.py --synthetic (they override the defaults in CrowdES/evaluate_synthetic_dataset.py)
GEN_ARGS=(--dataset_root "${DATASET_ROOT}")
if [ -n "${SCENE_NAME}" ]; then
    GEN_ARGS+=(--scene "${SCENE_NAME}")
elif [ -n "${TOWN_FILTER}" ]; then
    GEN_ARGS+=(--town "${TOWN_FILTER}")
fi
if [ -n "${OUT_SUBSET}" ]; then
    GEN_ARGS+=(--out_subset "${OUT_SUBSET}")
fi
if [ "${RANDOMIZE}" == "true" ]; then
    GEN_ARGS+=(--randomize)
fi
if [ "${FORCE}" == "true" ]; then
    GEN_ARGS+=(--force)
fi


# Use the active Python environment (see Installation in README.md)
if ! python -c "import torch, diffusers" > /dev/null 2>&1; then
    echo "❌ ERROR: PyTorch or diffusers is missing in the active Python environment ($(command -v python || echo 'no python on PATH'))."
    echo "   Activate the environment from the README first, e.g. conda activate topourban."
    exit 1
fi

# Run trajectory generation
echo ""
echo "Starting trajectory generation..."
echo ""
python trainval.py \
    --synthetic \
    --model_config "${MODEL_CONFIG}" \
    "${GEN_ARGS[@]}"

echo ""
echo "======================================"
echo "✅ Trajectory generation completed!"
echo "======================================"
echo ""
echo "Output location: ${PROJECT_ROOT}/output/generated/synthetic/${OUT_SUBSET:-<standard_ft|topo_ft>}/"
echo ""
echo "Generated files:"
echo "  - *-crowdes-synthetic-*.csv         (Trajectory data for IL training)"
echo "  - *-crowdes-synthetic-*_last_frame.png  (Expert trajectory visualization)"
echo "  - *_group_graph.json                (Group formation topology)"
echo ""
