#!/bin/bash
# Generate CARLA scenarios with queue/gathering formations
# Usage: ./scripts/generate_carla_scenarios.sh [--town TOWN] [--num_images N] [--dataset_root PATH]
# Needs CARLA 0.9.16 in ../CARLA_0.9.16, next to the repository, or set CARLA_ROOT to its folder.
#
# Examples:
#   ./scripts/generate_carla_scenarios.sh                              # Town10HD_Opt, 30 scenes
#   ./scripts/generate_carla_scenarios.sh --town Town05 --num_images 15
#   ./scripts/generate_carla_scenarios.sh --town Town03 --num_images 10

set -e

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CROWDES_DIR="${PROJECT_ROOT}/topourbanbench/generation/crowdes"

# Default parameters
TOWN="Town10HD_Opt"
NUM_IMAGES=30
DATASET_ROOT="${CROWDES_DIR}/datasets/Synthetic"
START_SCENE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --town)         TOWN="$2";         shift 2 ;;
        --num_images)   NUM_IMAGES="$2";   shift 2 ;;
        --dataset_root) DATASET_ROOT="$2"; shift 2 ;;
        --start_scene)  START_SCENE="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================"
echo "CARLA Scenario Generation"
echo "======================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "CrowdES Dir:  ${CROWDES_DIR}"
echo "Town:         ${TOWN}"
echo "Num Images:   ${NUM_IMAGES}"
echo "Start Scene:  ${START_SCENE}"
echo "Dataset Root: ${DATASET_ROOT}"
echo "======================================"

# Kill existing CARLA server if running
if pgrep -f "CarlaUE4" > /dev/null; then
    echo ""
    echo "Killing existing CARLA server..."
    pkill -f "CarlaUE4"
    sleep 3
fi

# Start CARLA server
CARLA_PATH="${CARLA_ROOT:-${PROJECT_ROOT}/../CARLA_0.9.16}"
if [ ! -f "${CARLA_PATH}/CarlaUE4.sh" ]; then
    echo "❌ ERROR: CARLA not found at ${CARLA_PATH}"
    echo "   Extract CARLA 0.9.16 there, or set CARLA_ROOT to the folder that contains CarlaUE4.sh."
    exit 1
fi

echo ""
echo "Starting CARLA server..."
cd "${CARLA_PATH}"
# setsid gives the server its own process group, so stopping the group also stops the simulator that CarlaUE4.sh starts
DISPLAY= setsid ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 > /tmp/carla_server.log 2>&1 &
CARLA_PID=$!
echo "Started CARLA server (PID: ${CARLA_PID})"
# Stop the server when the script ends, also after an error or Ctrl-C
trap 'echo "Stopping CARLA server..."; kill -- -${CARLA_PID} 2>/dev/null || true' EXIT

echo "Waiting for CARLA server to initialize..."
sleep 20

cd "${PROJECT_ROOT}"

# Navigate to crowdes directory (the Python environment from the README must be active)
cd "${CROWDES_DIR}"

# Run the scenario generation script
python extract_carla_maps.py \
    --town "${TOWN}" \
    --num_images "${NUM_IMAGES}" \
    --start_scene "${START_SCENE}" \
    --dataset_root "${DATASET_ROOT}"

echo ""
echo "✅ Scenario generation completed!"
echo "Town: ${TOWN} | Scenes: ${NUM_IMAGES}"
echo "Output location: ${DATASET_ROOT}"
echo ""
echo "Generated files:"
echo "  - image_terrain/*.png      (Semantic maps)"
echo "  - gt/*.png                  (Walkable maps, scenario maps)"
echo "  - metadata/*.json           (Queue/gathering metadata)"
echo "  - homography/*.txt          (Homography matrices)"
