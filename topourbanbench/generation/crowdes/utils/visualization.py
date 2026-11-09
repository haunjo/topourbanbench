import os
import cv2
import numpy as np
from tqdm.auto import tqdm

group_color_dict = {
        'queue':      (255, 50,  50),   # red
        'gathering':  (50,  255, 50),   # green
        'cohesive':   (50,  150, 255),  # blue
        'crosswalk':  (255, 200, 50),   # yellow
        'ungrouped':  (255, 255, 255),  # white
    }

# Semi-transparent overlay alpha per group type
GATHERING_ALPHA  = 0.25   # gathering space fill (O-space)
COHESIVE_ALPHA   = 0.35   # cohesive buffer zone fill (R-space)
TOPOLOGY_ALPHA   = 0.55   # line opacity (drawn directly, not blended)


def draw_filled_circle(image, center, size, fill_color, edge_color, line_thickness):
    half_size = int(size // 2)
    cv2.circle(image, center, half_size, fill_color, -1)
    cv2.circle(image, center, half_size, edge_color, line_thickness)


def draw_filled_triangle(image, center, size, fill_color, edge_color, line_thickness):
    height   = size * (np.sqrt(3) / 2)
    half_size = size / 2
    vertices = np.array([
        [center[0],              center[1] - height / 3 * 2],
        [center[0] - half_size,  center[1] + height / 3],
        [center[0] + half_size,  center[1] + height / 3],
    ], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [vertices], fill_color)
    cv2.polylines(image, [vertices], isClosed=True, color=edge_color, thickness=line_thickness)


def draw_filled_square(image, center, size, fill_color, edge_color, line_thickness):
    half_size = size // 2
    vertices = np.array([
        [center[0] - half_size, center[1] - half_size],
        [center[0] + half_size, center[1] - half_size],
        [center[0] + half_size, center[1] + half_size],
        [center[0] - half_size, center[1] + half_size],
    ], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [vertices], fill_color)
    cv2.polylines(image, [vertices], isClosed=True, color=edge_color, thickness=line_thickness)


def _nearest_neighbor_chain(positions):
    """Return positions reordered as a nearest-neighbor chain (for queue lines)."""
    pts = [tuple(p) for p in positions]
    if len(pts) <= 1:
        return positions
    remaining = list(range(1, len(pts)))
    order     = [0]
    while remaining:
        last = order[-1]
        dists = [np.linalg.norm(positions[i] - positions[last]) for i in remaining]
        nxt   = remaining[int(np.argmin(dists))]
        order.append(nxt)
        remaining.remove(nxt)
    return positions[order]


def _draw_group_topology(frame_img, frame_agents, line_thickness):
    """
    Draw group topology overlays on frame_img (in-place):

    Space visualization (semi-transparent fills):
    · gathering  — O-space: donut shape showing transactional core
                   (inner circle passable, outer ring is social boundary)
    · cohesive   — R-space: buffer zone around convex hull (D_buffer ≈ 1.2m)
                   visualizes the CPI metric's protected zone

    Topology lines (P-space):
    · cohesive   — clique lines between all member pairs
    · queue      — chain lines along nearest-neighbor order
    · crosswalk  — clique lines between all member pairs (disabled)
    """
    grouped_agents = frame_agents[frame_agents['group_id'] >= 0]
    if grouped_agents.empty:
        return

    # ── Pass 1: gathering O-space + cohesive R-space fills ────────────────────
    overlay_gathering = frame_img.copy()
    overlay_cohesive  = frame_img.copy()
    has_gathering = False
    has_cohesive  = False

    for gid, members in grouped_agents.groupby('group_id'):
        g_type = str(members['group_type'].iloc[0])
        positions = members[['x', 'y']].values.astype(float)

        if g_type == 'gathering':
            # O-space: donut shape around gathering center
            center = positions.mean(axis=0)
            dists  = np.linalg.norm(positions - center, axis=1)

            # Outer radius: mean + 1σ gives a comfortable boundary around the ring
            outer_r = int(dists.mean() + dists.std() + 15) if len(dists) > 1 else 80
            inner_r = max(0, int(dists.mean() - dists.std() - 10))

            color = group_color_dict['gathering']
            cx, cy = int(round(center[0])), int(round(center[1]))

            # Filled outer circle
            cv2.circle(overlay_gathering, (cx, cy), outer_r, color, -1)
            # Carve out inner region (show the robot-passable center)
            if inner_r > 0:
                cv2.circle(overlay_gathering, (cx, cy), inner_r, (0, 0, 0), -1)

            has_gathering = True

        elif g_type == 'cohesive':
            # R-space: buffer zone around convex hull (for CPI visualization)
            buffer_px = 24  # ~1.2m at typical scale
            color = group_color_dict['cohesive']
            mask = np.zeros(frame_img.shape[:2], dtype=np.uint8)

            if len(positions) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(positions)
                    hull_points = positions[hull.vertices].astype(np.int32)
                    cv2.fillPoly(mask, [hull_points], 255)
                except Exception:
                    # Degenerate hull — fall through to capsule path
                    for p in positions.astype(np.int32):
                        cv2.circle(mask, tuple(p), buffer_px // 2, 255, -1)

            if len(positions) == 2:
                # Capsule: filled line with round caps between the two members
                p0 = tuple(positions[0].astype(np.int32))
                p1 = tuple(positions[1].astype(np.int32))
                cv2.line(mask, p0, p1, 255, buffer_px)
                cv2.circle(mask, p0, buffer_px // 2, 255, -1)
                cv2.circle(mask, p1, buffer_px // 2, 255, -1)

            if len(positions) == 1:
                cv2.circle(mask, tuple(positions[0].astype(np.int32)), buffer_px, 255, -1)

            if mask.any():
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (buffer_px * 2, buffer_px * 2))
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                for c in range(3):
                    overlay_cohesive[:, :, c][mask_dilated > 0] = color[c]
                has_cohesive = True

    # Apply gathering overlay first (darker)
    if has_gathering:
        cv2.addWeighted(overlay_gathering, GATHERING_ALPHA, frame_img,
                       1.0 - GATHERING_ALPHA, 0, frame_img)

    # Apply cohesive overlay (lighter, so it doesn't overwhelm the lines)
    if has_cohesive:
        cv2.addWeighted(overlay_cohesive, COHESIVE_ALPHA, frame_img,
                       1.0 - COHESIVE_ALPHA, 0, frame_img)

    # ── Pass 2: topology lines ────────────────────────────────────────────────
    topo_line_t = max(1, line_thickness + 1)

    for gid, members in grouped_agents.groupby('group_id'):
        g_type    = str(members['group_type'].iloc[0])
        color_bgr = group_color_dict.get(g_type, group_color_dict['ungrouped'])
        positions = members[['x', 'y']].values.astype(int)
        N         = len(positions)

        if g_type in ('cohesive', 'crosswalk'):
            # Clique: all-pairs edges
            for i in range(N):
                for j in range(i + 1, N):
                    cv2.line(frame_img,
                             tuple(positions[i]), tuple(positions[j]),
                             color_bgr, topo_line_t, cv2.LINE_AA)

        elif g_type == 'queue':
            # Chain: sorted by nearest-neighbor traversal
            if N >= 2:
                ordered = _nearest_neighbor_chain(positions)
                for k in range(len(ordered) - 1):
                    cv2.line(frame_img,
                             tuple(ordered[k]), tuple(ordered[k + 1]),
                             color_bgr, topo_line_t, cv2.LINE_AA)

        elif g_type == 'gathering':
            # Draw a thin circle outline at the mean radius for clarity
            center = members[['x', 'y']].values.astype(float).mean(axis=0)
            dists  = np.linalg.norm(members[['x', 'y']].values - center, axis=1)
            mean_r = int(dists.mean()) if len(dists) > 0 else 80
            cv2.circle(frame_img,
                       (int(round(center[0])), int(round(center[1]))),
                       mean_r, color_bgr, topo_line_t, cv2.LINE_AA)


def produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario,
                            scenario_length, config, save_last_frame=True):
    """
    Produce a video from the generated scenario data.

    Draws each frame at native resolution (2048×2048) with group topology
    overlays, then downscales to half resolution (1024×1024) before writing.

    Params:
        video_path (str): Path to save the video.
        scene_img (np.ndarray): The background image of the scene.
        scene_bg (np.ndarray): The background image of the scene.
        generated_scenario (pd.DataFrame): DataFrame containing the generated scenario data.
        scenario_length (int): Length of the scenario in frames.
        config (Config): Configuration object containing simulation parameters.
        save_last_frame (bool): If True, save the last frame as an image. Default: True.
    """

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    fill_colors = [
        (234, 56, 41), (246, 133, 17), (248, 217, 4),  (170, 216, 22),
        (39, 187, 54), (0,  143, 93),  (15, 181, 174), (51,  197, 232),
        (56, 146, 243),(104, 109, 244),(137,  61, 231), (224,  85, 226),
        (222,  61, 130),
    ]
    edge_color = (177, 177, 177)

    # Drawing constants at native resolution
    native_h, native_w = scene_img.shape[0], scene_img.shape[1]
    ref            = max(native_h, native_w)
    line_thickness = max(1, round(ref / 720 * 1))
    circle_size    = max(8,  round(ref / 720 * 12))   # 20 → 12
    triangle_size  = max(8,  round(ref / 720 * 12))   # 20 → 12
    square_size    = max(6,  round(ref / 720 * 10))   # 16 → 10
    font_scale     = max(0.25, round(ref / 720 * 0.35))  # 0.5 → 0.35
    font_thickness = max(1,   round(ref / 720 * 1.5))    # 2 → 1.5

    # Video at half resolution
    video_w, video_h = native_w // 2, native_h // 2
    fourcc       = cv2.VideoWriter_fourcc(*'XVID')
    fps          = config.crowd_simulator.simulator.simulator_fps
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (video_w, video_h))

    frame_map     = {f: grp for f, grp in generated_scenario.groupby('frame')}
    dataset_fps   = config.dataset.dataset_fps
    simulator_fps = config.crowd_simulator.simulator.simulator_fps
    frame_stride  = dataset_fps // simulator_fps

    bg_bgr = cv2.cvtColor(np.array(scene_bg), cv2.COLOR_RGB2BGR)

    last_frame_img = None
    for frame in tqdm(range(scenario_length), desc='Generating video'):
        if frame % frame_stride != 0:
            continue

        frame_img    = bg_bgr.copy()
        frame_agents = frame_map.get(frame)

        if frame_agents is not None:
            # ── 1. Group topology overlays (behind agents) ────────────────────
            _draw_group_topology(frame_img, frame_agents, line_thickness)

            # ── 2. Individual agents ──────────────────────────────────────────
            for _, agent in frame_agents.iterrows():
                agent_id    = int(agent['agent_id'])
                agent_type  = int(agent['agent_type'])
                agent_x     = int(agent['x'])
                agent_y     = int(agent['y'])
                agent_color = fill_colors[agent_id % len(fill_colors)][::-1]
                group_type  = agent.get('group_type', 'none')
                group_id    = agent.get('group_id', -1)

                is_expert = (group_type == 'expert')
                if is_expert:
                    draw_filled_circle(frame_img, (agent_x, agent_y), circle_size,
                                       (0, 0, 0), (255, 255, 255), line_thickness + 1)
                elif agent_type == 0:
                    draw_filled_circle(frame_img, (agent_x, agent_y), circle_size,
                                       agent_color, edge_color, line_thickness)
                elif agent_type == 1:
                    draw_filled_triangle(frame_img, (agent_x, agent_y), triangle_size,
                                         agent_color, edge_color, line_thickness)
                elif agent_type == 2:
                    draw_filled_square(frame_img, (agent_x, agent_y), square_size,
                                       agent_color, edge_color, line_thickness)

                if is_expert:
                    text        = 'e'
                    label_color = (255, 255, 255)
                elif group_id >= 0:
                    text        = str(int(group_id))
                    label_color = group_color_dict.get(group_type, group_color_dict['ungrouped'])
                else:
                    text = None

                if text is not None:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale, font_thickness)[0]
                    tx = agent_x - text_size[0] // 2
                    ty = agent_y + text_size[1] // 2
                    cv2.putText(frame_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                    cv2.putText(frame_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, label_color, font_thickness, cv2.LINE_AA)

            cv2.putText(frame_img, f'Population: {len(frame_agents)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_small = cv2.resize(frame_img, (video_w, video_h), interpolation=cv2.INTER_AREA)
        video_writer.write(frame_small)
        last_frame_img = frame_img

    video_writer.release()

    if save_last_frame and last_frame_img is not None:
        last_frame_path = video_path.replace('.avi', '_last_frame.png')
        cv2.imwrite(last_frame_path, last_frame_img)
        print(f'Last frame saved at {last_frame_path}')


def save_expert_trajectory_png(output_path, scene_bg, generated_scenario):
    """
    Draw all agent positions at the last frame + expert agent full trajectories,
    then save as a single PNG.

    Expert agents are drawn as full trajectory polylines (one color per agent).
    Crowd agents are drawn at their last-frame position only.

    Args:
        output_path (str): Path to save the PNG (e.g., '.../{scene}-crowdes-synthetic-0_last_frame.png')
        scene_bg (np.ndarray): Background image (RGB).
        generated_scenario (pd.DataFrame): Columns: agent_id, agent_type, frame, x, y, group_id, group_type
    """
    expert_colors = [
        (0,   200, 255),  # cyan
        (255, 100,   0),  # orange
        (180,   0, 255),  # purple
        (0,   255, 128),  # green
        (255,   0, 100),  # pink
        (255, 220,   0),  # yellow
    ]

    bg_bgr = cv2.cvtColor(np.array(scene_bg), cv2.COLOR_RGB2BGR)
    img = bg_bgr.copy()

    native_h, native_w = img.shape[:2]
    ref = max(native_h, native_w)
    circle_size    = max(6,  round(ref / 720 * 10))
    line_thickness = max(2,  round(ref / 720 * 3))
    font_scale     = max(0.4, round(ref / 720 * 0.5))
    font_thickness = max(1,   round(ref / 720 * 2))

    # ── Crowd agents: draw at last frame position ──────────────────────────
    fill_colors = [
        (234, 56, 41), (246, 133, 17), (248, 217, 4),  (170, 216, 22),
        (39, 187, 54), (0,  143, 93),  (15, 181, 174), (51,  197, 232),
        (56, 146, 243),(104, 109, 244),(137,  61, 231), (224,  85, 226),
        (222,  61, 130),
    ]
    edge_color = (177, 177, 177)

    crowd_df = generated_scenario[generated_scenario['group_type'] != 'expert']
    if not crowd_df.empty:
        last_frame = crowd_df['frame'].max()
        last_agents = crowd_df[crowd_df['frame'] == last_frame]
        _draw_group_topology(img, last_agents, max(1, line_thickness - 1))
        for _, agent in last_agents.iterrows():
            aid   = int(agent['agent_id'])
            atype = int(agent['agent_type'])
            ax, ay = int(agent['x']), int(agent['y'])
            color = fill_colors[aid % len(fill_colors)][::-1]
            if atype == 0:
                draw_filled_circle(img, (ax, ay), circle_size, color, edge_color, 1)
            elif atype == 1:
                draw_filled_triangle(img, (ax, ay), circle_size, color, edge_color, 1)
            elif atype == 2:
                draw_filled_square(img, (ax, ay), circle_size, color, edge_color, 1)

    # ── Expert agents: draw full trajectory polylines ──────────────────────
    expert_df = generated_scenario[generated_scenario['group_type'] == 'expert']
    expert_ids = sorted(expert_df['agent_id'].unique())

    for color_idx, eid in enumerate(expert_ids):
        traj = expert_df[expert_df['agent_id'] == eid].sort_values('frame')
        pts  = traj[['x', 'y']].values.astype(np.int32)
        if len(pts) < 2:
            continue
        color = expert_colors[color_idx % len(expert_colors)]
        color_bgr = color[::-1]

        # Draw trajectory line
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color_bgr, line_thickness, cv2.LINE_AA)

        # Mark start (hollow circle) and end (filled circle)
        cv2.circle(img, tuple(pts[0]),  circle_size, color_bgr, 2, cv2.LINE_AA)
        draw_filled_circle(img, tuple(pts[-1]), circle_size, color_bgr, (0, 0, 0), 2)

        # Label start with agent id
        label = f'e{eid}'
        cv2.putText(img, label, (pts[0][0] + 8, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(img, label, (pts[0][0] + 8, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, font_thickness, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f'[ExpertTraj] Saved to {output_path}')
