"""
Description. A simple programmatic solution to the up-down skiing problem.
"""
summit_height = 3900
base_height = 1600
down_dist = 400
up_dist = 200

distance_covered = 0
current_height = summit_height
ski_sessions = 0
while True:
    # Ski down
    down_ski_dist = min(down_dist, current_height - base_height)
    current_height -= down_ski_dist
    distance_covered += down_ski_dist
    ski_sessions += 1

    if current_height <= base_height:
        print(f"Down at base, total ski distance: {distance_covered}, total ski sessions: {ski_sessions}")
        break

    # Ski up
    current_height += up_dist
