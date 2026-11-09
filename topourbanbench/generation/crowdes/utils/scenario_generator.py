import yaml
import cv2
import numpy as np


class SemanticPreprocessor:
    """
    Converts CARLA semantic segmentation into the base maps that the CrowdES pipeline needs.

    Maps produced:
    - walkable_mask: walkable area for the navigation mesh
    - appearance_map: spawn density map for the emitter (border regions)

    Scope: scenario-agnostic base maps only.
    Scenario-specific maps (queue, cohesive, etc.) are handled by UnifiedScenarioGenerator.
    """

    def __init__(self, config_path="scenario_config.yaml"):
        # 1. Load the YAML config and fix the random seed
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        np.random.seed(self.config.get('random_seed', 42))

        # 2. Read the semantic ID mapping from the YAML (with defaults)
        sem_cfg = self.config.get('semantic_ids', {})

        self.ROAD_ID = sem_cfg.get('road', 1)
        self.SIDEWALK_ID = sem_cfg.get('sidewalk', 2)
        self.ROADLINE_ID = sem_cfg.get('roadline', 24)
        self.BUILDING_ID = sem_cfg.get('building', 11)

        self.OVERHANG_IDS = sem_cfg.get('overhang_ids', [5, 6, 7, 20, 0, 22])
        self.VEGETATION_ID = sem_cfg.get('vegetation', 9)
        self.FENCE_ID = sem_cfg.get('fence', 4)


    def generate_base_maps(self, semantic_raw, pixel_per_meter=40.96, skip_cc_filter=False):
        """
        Takes raw CARLA semantic data and returns the walkable map for the NavMesh
        and the base appearance map for the emitter.

        Args:
            semantic_raw: CARLA semantic segmentation map
            pixel_per_meter: Scale factor for pixel-to-meter conversion (default: 40.96)

        IMPORTANT: Ensures appearance and scenario maps only use the largest connected
        component to prevent agents from spawning on disconnected islands.

        Args:
            semantic_raw: (H, W) CARLA semantic segmentation (class IDs)

        Returns:
            walkable_mask: (H, W) uint8 [0, 255], walkable areas (largest component only)
            appearance_map: (H, W) uint8 [0, 255], spawn density (borders)
        """
        # ==========================================
        # STEP 1: Base map (walkable map)
        # ==========================================

        valid_sidewalk_ids = [self.SIDEWALK_ID] + self.OVERHANG_IDS
        sidewalk = np.isin(semantic_raw, valid_sidewalk_ids).astype(np.uint8) * 255
        road = np.where(semantic_raw == self.ROAD_ID, 255, 0).astype(np.uint8)

        # ---------------------------------------------------------
        # 1. Road safety trim: remove sidewalk pixels next to the curb
        # Crosswalks are added later in extract_carla_maps.py with fillPoly, from CARLA Map API coordinates.
        # ---------------------------------------------------------
        curb_margin_meters = self.config.get('road_curb_margin_meters', 0.5)
        curb_margin_px = int(curb_margin_meters * pixel_per_meter)
        if curb_margin_px % 2 == 0:
            curb_margin_px += 1
        kernel_margin = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                  (curb_margin_px, curb_margin_px))
        dilated_road = cv2.dilate(road, kernel_margin)
        walkable = cv2.subtract(sidewalk, dilated_road)

        # ---------------------------------------------------------
        # 4. Global smoothing that keeps the usable area (avoids isolated pieces)
        # ---------------------------------------------------------
        final_smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        walkable = cv2.morphologyEx(walkable, cv2.MORPH_OPEN, final_smooth_kernel)
        walkable = cv2.morphologyEx(walkable, cv2.MORPH_CLOSE, final_smooth_kernel)

        # With skip_cc_filter=True the connected-component filter is skipped here,
        # so that extract_carla_maps.py can run it after the crosswalks are filled in.
        if not skip_cc_filter:
            connectivity_cfg = self.config.get('walkable_connectivity', {})
            connectivity_mode = connectivity_cfg.get('mode', 'largest_only')
            min_area = connectivity_cfg.get('min_component_area', 1000)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walkable, connectivity=8)

            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                valid_walkable = np.zeros_like(walkable)

                if connectivity_mode == 'largest_only':
                    largest_component_idx = np.argmax(areas) + 1
                    valid_walkable[labels == largest_component_idx] = 255
                    if num_labels > 2:
                        discarded = len([a for a in areas if a >= min_area]) - 1
                        print(f"  [Connected Components] {num_labels-1} component(s), keeping largest, discarding {discarded}")

                elif connectivity_mode == 'all_valid':
                    num_kept = 0
                    for idx in range(1, num_labels):
                        area = stats[idx, cv2.CC_STAT_AREA]
                        if area >= min_area:
                            valid_walkable[labels == idx] = 255
                            num_kept += 1
                    if num_kept > 1:
                        print(f"  [Warning] all_valid mode: {num_kept} islands kept — cross-island spawning possible")

                walkable = valid_walkable
            else:
                print(f"  [Warning] No walkable areas found")
                walkable = np.zeros_like(walkable)

        # ==========================================
        # STEP 2: Base appearance map for the emitter
        # ==========================================
        appearance_map = self._generate_base_appearance(walkable)

        # ==========================================
        # STEP 3: Smooth the final probability map (Gaussian blur)
        # ==========================================
        sigma = self.config.get('probability_smoothing_sigma', 15)
        appearance_map = cv2.GaussianBlur(appearance_map, (sigma, sigma), 0)

        return walkable, appearance_map

    # ---------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------

    def _generate_base_appearance(self, walkable):
        """
        Uses an image border as an appearance region only where the walkable area touches it.
        Works for any road shape (straight, L, T, and so on).
        """
        appearance_cfg = self.config.get('appearance', {})
        thickness = appearance_cfg.get('border_thickness',
                                       self.config.get('appearance_border_thickness', 45))

        appearance = np.zeros_like(walkable)

        if walkable[:thickness, :].any():
            appearance[:thickness, :] = 255
        if walkable[-thickness:, :].any():
            appearance[-thickness:, :] = 255
        if walkable[:, :thickness].any():
            appearance[:, :thickness] = 255
        if walkable[:, -thickness:].any():
            appearance[:, -thickness:] = 255

        appearance = cv2.bitwise_and(walkable, appearance)

        if np.sum(appearance) == 0:
            print("[Appearance Map] No border walkable found, falling back to Canny edges")
            edges = cv2.Canny(walkable, 100, 200)
            appearance = cv2.dilate(edges, np.ones((15, 15), np.uint8))

        return appearance
