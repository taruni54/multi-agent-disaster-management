import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np, heapq, random
from scipy.ndimage import binary_dilation

# ── CONFIG ────────────────────────────────────────────────────────────────────
AREA      = 100
G         = 100
SCAN_R    = 4
DRONE_R   = 8
DRONE_SPD = 3.5
PAUSE_F   = 12
SPD       = {"scout": 2.5, "clearer": 6.0, "rescue": 3.0}
FPS       = 30
ROBOT_RADIUS = 1

AFTERSHOCK_PENALTY = 80
AFTERSHOCK_DANGER_R = 6

def shrink_block(x0, y0, x1, y1, scale=0.62):
    cx, cy = (x0+x1)/2, (y0+y1)/2
    w, h   = (x1-x0)*scale, (y1-y0)*scale
    return cx-w/2, cy-h/2, cx+w/2, cy+h/2

# ── GRIDS ─────────────────────────────────────────────────────────────────────
grid     = np.zeros((G, G), dtype=np.uint8)
nav_grid = np.zeros((G, G), dtype=np.uint8)
penalty_grid = np.zeros((G, G), dtype=np.float32)

def rebuild_nav_grid():
    global nav_grid
    struct   = np.ones((2*ROBOT_RADIUS+1, 2*ROBOT_RADIUS+1), dtype=bool)
    nav_grid = binary_dilation(grid.astype(bool), structure=struct).astype(np.uint8)

def rebuild_penalty_grid():
    global penalty_grid
    penalty_grid = np.zeros((G, G), dtype=np.float32)
    for a in AFTERSHOCKS:
        if not a["spawned"] or a["label"] in cleared_aftershocks:
            continue
        cx, cy = int(a["cx"]), int(a["cy"])
        r = AFTERSHOCK_DANGER_R
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    gx, gy = cx+dx, cy+dy
                    if 0 <= gx < G and 0 <= gy < G:
                        dist = np.hypot(dx, dy)
                        penalty_grid[gy, gx] += AFTERSHOCK_PENALTY * max(0, 1.0 - dist/r)

def block(x0, y0, x1, y1):
    grid[max(0,int(y0)):min(G,int(y1)), max(0,int(x0)):min(G,int(x1))] = 1

block(0,0,100,2); block(0,98,100,100)
block(0,0,2,100); block(98,0,100,100)

# ── RUBBLE ────────────────────────────────────────────────────────────────────
RAW_RUBBLE = [
    (6,77,32,81,"SLAB-T"),(6,63,32,67,"SLAB-B"),(6,67,11,77,"WALL-L"),(27,67,33,77,"WALL-R"),
    (62,83,90,87,"ROOF-T"),(62,87,68,97,"SIDE-L"),(84,87,90,97,"SIDE-R"),
    (44,65,62,69,"CAP-T"),(44,56,62,60,"CAP-B"),(44,60,49,65,"CAP-L"),(57,60,62,65,"CAP-R"),
    (78,55,95,58,"BOX-T"),(78,63,95,66,"BOX-B"),(78,58,83,63,"BOX-L"),(90,58,95,63,"BOX-R"),
    (18,50,34,53,"PEN-T"),(18,59,34,62,"PEN-B"),(18,53,22,59,"PEN-L"),(30,53,34,59,"PEN-R"),
    (62,10,78,13,"CAGE-T"),(62,18,78,21,"CAGE-B"),(62,13,66,18,"CAGE-L"),(74,13,78,18,"CAGE-R"),
    (43,38,48,43,"WALL-C2"),(52,54,64,58,"HEAP-1"),
    (67,38,78,44,"HEAP-2"),(72,68,84,72,"HEAP-3"),(18,44,32,48,"CHUNK-A"),
    (34,83,56,87,"CHUNK-B"),(56,73,68,77,"CHUNK-C"),(80,46,92,50,"CHUNK-D"),
    (22,64,36,68,"CHUNK-E"),(62,26,72,32,"DEBRIS-1"),(28,22,38,28,"DEBRIS-2"),
    (10,28,20,34,"SHARD-1"),(84,28,92,36,"SHARD-2"),
    (34,52,38,64,"PILLAR-1"),(48,72,54,82,"PILLAR-2"),(68,46,74,54,"PILLAR-3"),
    (14,86,26,92,"PILLAR-4"),(58,10,62,16,"SLAB-F1"),(82,14,92,20,"SLAB-F2"),
    (6,14,16,20,"SLAB-F3"),(36,14,46,20,"SLAB-F4"),(88,68,96,74,"WALL-R1"),
    (4,40,10,52,"WALL-L1"),(50,30,56,38,"POST-1"),(74,76,80,84,"POST-2"),
    (26,76,32,84,"POST-3"),(88,80,96,88,"CORNER-R"),(4,78,10,86,"CORNER-L"),
]

RUBBLE = []
for x0,y0,x1,y1,lbl in RAW_RUBBLE:
    nx0,ny0,nx1,ny1 = shrink_block(x0,y0,x1,y1, scale=0.62)
    RUBBLE.append((nx0,ny0,nx1,ny1,lbl))
    block(nx0,ny0,nx1,ny1)

grid[48:55, 38:43] = 0
grid[38:44, 38:43] = 0

rebuild_nav_grid()

# ── AFTERSHOCKS — RANDOMLY GENERATED ─────────────────────────────────────────
def _gen_random_aftershocks(n=3):
    """Generate n aftershocks with random positions, sizes, spawn frames and rumble params."""
    rng = random.Random()
    BASE_EXCLUSION = (38, 0, 62, 8)

    spawn_frames = sorted(rng.sample(range(300, 900), n))
    result = []
    for i, sf in enumerate(spawn_frames):
        while True:
            cx = rng.uniform(10, 90)
            cy = rng.uniform(10, 95)
            if BASE_EXCLUSION[0] <= cx <= BASE_EXCLUSION[2] and cy <= BASE_EXCLUSION[3]:
                continue
            if all(np.hypot(cx - r["cx"], cy - r["cy"]) >= 18 for r in result):
                break

        hw = rng.uniform(2.5, 4.5)
        hh = rng.uniform(2.0, 3.5)
        angle = rng.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)
        spd = rng.uniform(0.06, 0.11)

        result.append({
            "cx": cx, "cy": cy,
            "hw": hw, "hh": hh,
            "spawn_frame": sf,
            "label": f"AFTER-{i+1}",
            "rumble_speed": spd,
            "rumble_dir": (dx, dy),
        })
    return result

AFTERSHOCKS = _gen_random_aftershocks(3)
for a in AFTERSHOCKS:
    a.update(spawned=False, patch=None, lbl_art=None, warn_art=None,
             blocked_victim=None, rumble_phase=0.0,
             orig_cx=a["cx"], orig_cy=a["cy"])

cleared_aftershocks = set()

# ── MODIFIED A* WITH DYNAMIC AFTERSHOCK PENALTY ──────────────────────────────
def nearest_free(gx, gy):
    gx = int(np.clip(gx,0,G-1)); gy = int(np.clip(gy,0,G-1))
    if nav_grid[gy,gx]==0: return gx,gy
    for r in range(1,40):
        for dx in range(-r,r+1):
            for dy in range(-r,r+1):
                if abs(dx)!=r and abs(dy)!=r: continue
                nx,ny=gx+dx,gy+dy
                if 0<=nx<G and 0<=ny<G and nav_grid[ny,nx]==0: return nx,ny
    return gx,gy


def astar(start, goal, use_penalty=True):
    sx,sy = nearest_free(int(start[0]),int(start[1]))
    gx,gy = nearest_free(int(goal[0]), int(goal[1]))
    if (sx,sy)==(gx,gy):
        return [(float(sx)+0.5, float(sy)+0.5)]

    open_set = [(0,(sx,sy))]
    came = {}
    g_cost = {(sx,sy):0.0}

    while open_set:
        _,cur = heapq.heappop(open_set)
        cx,cy = cur
        if (cx,cy)==(gx,gy):
            path=[]
            while cur in came:
                path.append((cur[0]+0.5,cur[1]+0.5)); cur=came[cur]
            return path[::-1] or [(float(gx)+0.5,float(gy)+0.5)]

        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx,ny=cx+dx,cy+dy
            if not(0<=nx<G and 0<=ny<G): continue
            if nav_grid[ny,nx]==1: continue
            if dx!=0 and dy!=0:
                if nav_grid[cy,nx]==1 or nav_grid[ny,cx]==1: continue

            move_cost = 1.414 if (dx and dy) else 1.0
            if use_penalty:
                move_cost += penalty_grid[ny, nx]

            nb=(nx,ny)
            new_g = g_cost[cur] + move_cost
            if new_g < g_cost.get(nb, 1e18):
                came[nb]=cur
                g_cost[nb]=new_g
                h = abs(nx-gx)+abs(ny-gy)
                heapq.heappush(open_set,(new_g+h, nb))
    return []


def path_exists(s, g):
    return len(astar(s, g)) > 0


def set_path(robot, goal, use_penalty=True):
    p = astar((robot["x"],robot["y"]), goal, use_penalty=use_penalty)
    if p: robot["path"]=p; robot["pi"]=0


def move(robot, spd):
    if robot["pi"] >= len(robot["path"]): return True
    tx,ty = robot["path"][robot["pi"]]
    gx_t,gy_t = int(tx), int(ty)
    if 0<=gx_t<G and 0<=gy_t<G and nav_grid[gy_t,gx_t]==1:
        robot["path"]=[]; robot["pi"]=0; return True
    dx,dy = tx-robot["x"], ty-robot["y"]
    d = np.hypot(dx,dy)
    if d < spd:
        robot["x"],robot["y"] = tx,ty
        robot["pi"] += 1
    else:
        nx = robot["x"] + dx/d*spd
        ny = robot["y"] + dy/d*spd
        if nav_grid[int(np.clip(ny,0,G-1)), int(np.clip(nx,0,G-1))] == 0:
            robot["x"],robot["y"] = nx,ny
    return robot["pi"] >= len(robot["path"])


def move_drone(dr):
    if dr["wi"] >= len(dr["wps"]): return True
    tx,ty=dr["wps"][dr["wi"]]; dx,dy=tx-dr["x"],ty-dr["y"]; d=np.hypot(dx,dy)
    if d<DRONE_SPD: dr["x"],dr["y"]=tx,ty; dr["wi"]+=1
    else:           dr["x"]+=dx/d*DRONE_SPD; dr["y"]+=dy/d*DRONE_SPD
    return dr["wi"]>=len(dr["wps"])


def find_approach(cx, cy, hw, hh, offset=3):
    candidates = [
        (int(cx),       int(cy)-int(hh)-offset),
        (int(cx),       int(cy)+int(hh)+offset),
        (int(cx)-int(hw)-offset, int(cy)),
        (int(cx)+int(hw)+offset, int(cy)),
        (int(cx)-int(hw)-offset, int(cy)-int(hh)-offset),
        (int(cx)+int(hw)+offset, int(cy)+int(hh)+offset),
        (int(cx)-int(hw)-offset, int(cy)+int(hh)+offset),
        (int(cx)+int(hw)+offset, int(cy)-int(hh)-offset),
    ]
    return [nearest_free(px, py) for px, py in candidates]

# ── PRIORITY ──────────────────────────────────────────────────────────────────
PRIO_ORDER = {"CRITICAL":0,"MODERATE":1,"STABLE":2}
def prio_key(v): return PRIO_ORDER.get(v["priority"],99)

# ── VICTIMS ───────────────────────────────────────────────────────────────────
COLORS = {"CRITICAL":"#E63946","MODERATE":"#F4A261","STABLE":"#2DC653"}

victims = [
    {"id":"V-01","x":19.0,"y":72.0,"true_priority":"CRITICAL","true_thermal":38.7,"true_hr":148,"true_sound":"Moaning",  "blocked_by":[3], "ba":None},
    {"id":"V-02","x":74.0,"y":91.0,"true_priority":"CRITICAL","true_thermal":37.9,"true_hr":44, "true_sound":"None",    "blocked_by":[6], "ba":None},
    {"id":"V-07","x":52.0,"y":52.0,"true_priority":"CRITICAL","true_thermal":38.2,"true_hr":155,"true_sound":"Screaming","blocked_by":[], "ba":None},
    {"id":"V-11","x":85.0,"y":60.0,"true_priority":"CRITICAL","true_thermal":38.5,"true_hr":42, "true_sound":"Moaning",  "blocked_by":[14],"ba":None},
    {"id":"V-16","x":25.0,"y":55.0,"true_priority":"CRITICAL","true_thermal":38.6,"true_hr":151,"true_sound":"Moaning",  "blocked_by":[18],"ba":None},
    {"id":"V-18","x":68.0,"y":14.0,"true_priority":"CRITICAL","true_thermal":38.4,"true_hr":46, "true_sound":"Screaming","blocked_by":[22],"ba":None},
    {"id":"V-03","x":35.0,"y":45.0,"true_priority":"MODERATE","true_thermal":37.2,"true_hr":118,"true_sound":"Tapping",  "blocked_by":[], "ba":None},
    {"id":"V-04","x":82.0,"y":38.0,"true_priority":"STABLE",  "true_thermal":36.8,"true_hr":72, "true_sound":"Shouting", "blocked_by":[], "ba":None},
    {"id":"V-05","x":22.0,"y":30.0,"true_priority":"MODERATE","true_thermal":37.4,"true_hr":124,"true_sound":"Knocking", "blocked_by":[], "ba":None},
    {"id":"V-06","x":70.0,"y":74.0,"true_priority":"STABLE",  "true_thermal":36.6,"true_hr":68, "true_sound":"Talking",  "blocked_by":[], "ba":None},
    {"id":"V-08","x":13.0,"y":50.0,"true_priority":"MODERATE","true_thermal":37.1,"true_hr":112,"true_sound":"Tapping",  "blocked_by":[], "ba":None},
    {"id":"V-09","x":86.0,"y":42.0,"true_priority":"STABLE",  "true_thermal":36.9,"true_hr":76, "true_sound":"None",     "blocked_by":[], "ba":None},
    {"id":"V-10","x":44.0,"y":89.0,"true_priority":"MODERATE","true_thermal":37.3,"true_hr":109,"true_sound":"Knocking", "blocked_by":[], "ba":None},
    {"id":"V-12","x":30.0,"y":90.0,"true_priority":"STABLE",  "true_thermal":36.7,"true_hr":64, "true_sound":"Shouting", "blocked_by":[], "ba":None},
    {"id":"V-13","x":12.0,"y":38.0,"true_priority":"MODERATE","true_thermal":37.0,"true_hr":103,"true_sound":"Tapping",  "blocked_by":[], "ba":None},
    {"id":"V-14","x":86.0,"y":77.0,"true_priority":"STABLE",  "true_thermal":36.5,"true_hr":61, "true_sound":"None",     "blocked_by":[], "ba":None},
    {"id":"V-15","x":50.0,"y":24.0,"true_priority":"MODERATE","true_thermal":37.3,"true_hr":116,"true_sound":"Knocking", "blocked_by":[], "ba":None},
    {"id":"V-17","x":65.0,"y":55.0,"true_priority":"MODERATE","true_thermal":37.0,"true_hr":107,"true_sound":"Tapping",  "blocked_by":[], "ba":None},
    {"id":"V-19","x":48.0,"y":46.0,"true_priority":"STABLE",  "true_thermal":36.6,"true_hr":70, "true_sound":"Talking",  "blocked_by":[], "ba":None},
    {"id":"V-20","x":88.0,"y":24.0,"true_priority":"MODERATE","true_thermal":37.2,"true_hr":121,"true_sound":"Knocking", "blocked_by":[], "ba":None},
    {"id":"V-21","x":11.0,"y":81.0,"true_priority":"CRITICAL","true_thermal":38.8,"true_hr":152,"true_sound":"Screaming","blocked_by":[], "ba":None},
    {"id":"V-22","x":60.0,"y":36.0,"true_priority":"MODERATE","true_thermal":37.1,"true_hr":115,"true_sound":"Tapping",  "blocked_by":[], "ba":None},
    {"id":"V-23","x":34.0,"y":70.0,"true_priority":"STABLE",  "true_thermal":36.4,"true_hr":63, "true_sound":"Talking",  "blocked_by":[], "ba":None},
    {"id":"V-24","x":77.0,"y":18.0,"true_priority":"CRITICAL","true_thermal":38.3,"true_hr":47, "true_sound":"Moaning",  "blocked_by":[], "ba":None},
]

for v in victims:
    v.update(priority=None,thermal=None,hr=None,sound=None,
             detected=False,rescued=False,assigned=False,waiting=False,assessed=False)
    fx,fy = nearest_free(int(v["x"]),int(v["y"]))
    v["x"],v["y"] = float(fx)+0.5, float(fy)+0.5

BASE = (50,3)

# ── ROBOT FACTORIES ───────────────────────────────────────────────────────────
def make_scout(x,y,sid):
    fx,fy=nearest_free(int(x),int(y))
    return dict(x=float(fx)+0.5,y=float(fy)+0.5,path=[],pi=0,trail=[],
                status="scanning",pause_t=0,saved_path=[],saved_pi=0,
                id=sid,assess_target=None,stuck_timer=0,last_x=None,last_y=None)

def make_clearer(x,y,cid):
    fx,fy=nearest_free(int(x),int(y))
    return dict(x=float(fx)+0.5,y=float(fy)+0.5,path=[],pi=0,status="standby",task=None,timer=0,attempt_count=0,stuck_timer=0,last_x=None,last_y=None,id=cid)

def make_rescue(x,y,rid):
    fx,fy=nearest_free(int(x),int(y))
    return dict(x=float(fx)+0.5,y=float(fy)+0.5,path=[],pi=0,status="standby",target=None,timer=0,id=rid)

global_rescue_queue = []

# ── DRONES ────────────────────────────────────────────────────────────────────
def build_drone_path():
    wps=[]
    for i,x in enumerate(np.linspace(8,92,10)):
        wps+=[(x,95),(x,8)] if i%2==0 else [(x,8),(x,95)]
    return wps

def build_drone2_path():
    wps=[]
    for i,x in enumerate(np.linspace(92,8,10)):
        wps+=[(x,8),(x,95)] if i%2==0 else [(x,95),(x,8)]
    return wps

drone  = dict(x=50.,y=3., wps=build_drone_path(), wi=0,trail=[],status="flying",scan_flash=0,last_found=None,id="D1")
drone2 = dict(x=50.,y=97.,wps=build_drone2_path(),wi=0,trail=[],status="flying",scan_flash=0,last_found=None,id="D2")

# ── SCOUTS ────────────────────────────────────────────────────────────────────
SCOUT_DEFS = [
    (8,  6,  "S1"),
    (23, 93, "S2"),
    (38, 6,  "S3"),
    (53, 93, "S4"),
    (68, 6,  "S5"),
    (83, 93, "S6"),
]
scouts   = [make_scout(x,y,sid) for x,y,sid in SCOUT_DEFS]
clearers = [make_clearer(40,3,"C1"),make_clearer(47,3,"C2"),
            make_clearer(53,3,"C3"),make_clearer(60,3,"C4")]
rescues  = [make_rescue(44+i*4,3,f"R{i+1}") for i in range(5)]

global_clear_queue=[]; global_clear_in_prog=set(); cleared_rubble=set()
for _bi in [3,6,14,18,22]: global_clear_queue.append(("rubble",_bi))

# ── SCOUT COVERAGE PATH ───────────────────────────────────────────────────────
LANE_W   = 90/6
COL_STEP = SCAN_R * 1.6

def build_scout_coverage_path(scout_idx):
    s    = scouts[scout_idx]
    x_lo = 5 + scout_idx*LANE_W + 1.5
    x_hi = x_lo + LANE_W - 3.0
    go_up = (scout_idx % 2 == 0)
    cols = list(np.arange(x_lo, x_hi+COL_STEP, COL_STEP))
    cols = [min(c, x_hi) for c in cols]
    full_path = []
    cur_pos   = (s["x"], s["y"])
    for cx in cols:
        for wy in ([8.0,93.0] if go_up else [93.0,8.0]):
            fx,fy   = nearest_free(int(cx), int(wy))
            goal_f  = (fx+0.5, fy+0.5)
            seg     = astar(cur_pos, goal_f)
            if seg:
                full_path += seg
                cur_pos    = seg[-1]
        go_up = not go_up
    return full_path

for i in range(len(scouts)):
    scouts[i]["path"] = build_scout_coverage_path(i)

# ── STATE ─────────────────────────────────────────────────────────────────────
logs=[]; fnum=0; scanned=set()
v_dots=[]; v_lbls=[]; v_rings=[]
DETAIL_MAX=5; detected_log=[]; drone_marked_queue=[]; scout_assess_assigned=set()

def log(m):
    logs.append(m)
    if len(logs)>18: logs.pop(0)

# ── CHECK IF ALL VICTIMS DETECTED ─────────────────────────────────────────────
def all_victims_found():
    """Return True when all detected victims are RESCUED AND drones are done scanning."""
    # Only return true if drones have finished their scan
    if drone["status"] != "done" or drone2["status"] != "done":
        return False
    detected = [v for v in victims if v["detected"]]
    # All detected victims must be rescued, not just assessed
    return len(detected) > 0 and all(v.get("rescued") for v in detected)

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig,(ax,axp)=plt.subplots(1,2,figsize=(28,12),
                           gridspec_kw={"width_ratios":[1.8,1.4]})
fig.patch.set_facecolor("#030A12")
try: fig.canvas.manager.set_window_title("Disaster Sim — v14 (scouts return to base)")
except: pass

ax.set_facecolor("#050E1A"); ax.set_xlim(0,AREA); ax.set_ylim(0,AREA); ax.set_aspect("equal")
ax.set_title("DISASTER ZONE   ✈D1  ✈D2  ●SCOUT×6  ◆CLEARER×4  ■RESCUE×5  ⚡RANDOM AFTERSHOCK×3",
             color="#00E5FF",fontsize=8,fontweight="bold",fontfamily="monospace")
for sp in ax.spines.values(): sp.set_color("#1B3A5C")
ax.tick_params(colors="#2a4a5e",labelsize=6)
for i in range(0,101,10):
    ax.axvline(i,color="#1B3A5C",lw=0.15,alpha=0.18)
    ax.axhline(i,color="#1B3A5C",lw=0.15,alpha=0.18)
ax.add_patch(Rectangle((24,0),52,6,lw=1.5,edgecolor="#FFD166",facecolor="#0d2040",zorder=3))
ax.text(50,3,"BASE",color="#FFD166",ha="center",fontsize=7,fontweight="bold",fontfamily="monospace",zorder=4)

rubble_patches=[]
for x0,y0,x1,y1,lbl in RUBBLE:
    p=Rectangle((x0,y0),x1-x0,y1-y0,facecolor="#3a2208",edgecolor="#7a4010",lw=0.6,zorder=2,alpha=0.93)
    ax.add_patch(p); rubble_patches.append(p)
    if (x1-x0)>5 and (y1-y0)>3:
        ax.text((x0+x1)/2,(y0+y1)/2,lbl,color="#6a4520",ha="center",va="center",
                fontsize=3.0,fontfamily="monospace",zorder=3)

# Aftershock patches — built after random generation
for a in AFTERSHOCKS:
    a["warn_art"]=Circle((a["cx"],a["cy"]),max(a["hw"],a["hh"])+4,
                         color="#FF6600",fill=False,lw=1.0,ls=":",zorder=7,alpha=0)
    ax.add_patch(a["warn_art"])
    a["danger_circle"]=Circle((a["cx"],a["cy"]),AFTERSHOCK_DANGER_R,
                               color="#FF2200",fill=True,alpha=0,zorder=6,lw=0)
    ax.add_patch(a["danger_circle"])
    p=Rectangle((0,0),0,0,facecolor="#5a0000",edgecolor="#FF3300",lw=1.2,zorder=8,alpha=0)
    ax.add_patch(p); a["patch"]=p
    a["lbl_art"]=ax.text(a["cx"],a["cy"],"",color="#FF5555",fontsize=4,
                         ha="center",va="center",fontfamily="monospace",zorder=9)

sc_scat=ax.scatter([],[],s=4,color="#00B4D8",alpha=0.07,marker="s",zorder=1)
drone_trail_art,=ax.plot([],[],color="#AAFFE0",lw=0.5,alpha=0.18,zorder=5,ls=":")
drone_dot,=ax.plot([],[],"^",color="#AAFFE0",ms=11,zorder=12,markeredgecolor="#00FF99",markeredgewidth=1.2)
drone_ring=Circle((50,3),DRONE_R,color="#AAFFE0",fill=False,lw=0.8,ls="--",alpha=0.15,zorder=6); ax.add_patch(drone_ring)
drone_lbl=ax.text(50,6,"D1",color="#AAFFE0",fontsize=4.5,ha="center",fontfamily="monospace",zorder=13,fontweight="bold")
drone_flash=Circle((50,3),0,color="#AAFFE0",fill=True,alpha=0,zorder=11); ax.add_patch(drone_flash)
drone_beam,=ax.plot([],[],color="#AAFFE0",lw=0.8,ls=":",alpha=0,zorder=10)
drone2_trail_art,=ax.plot([],[],color="#FFD166",lw=0.5,alpha=0.18,zorder=5,ls=":")
drone2_dot,=ax.plot([],[],"^",color="#FFD166",ms=11,zorder=12,markeredgecolor="#FFAA00",markeredgewidth=1.2)
drone2_ring=Circle((50,97),DRONE_R,color="#FFD166",fill=False,lw=0.8,ls="--",alpha=0.15,zorder=6); ax.add_patch(drone2_ring)
drone2_lbl=ax.text(50,94,"D2",color="#FFD166",fontsize=4.5,ha="center",fontfamily="monospace",zorder=13,fontweight="bold")
drone2_flash=Circle((50,97),0,color="#FFD166",fill=True,alpha=0,zorder=11); ax.add_patch(drone2_flash)
drone2_beam,=ax.plot([],[],color="#FFD166",lw=0.8,ls=":",alpha=0,zorder=10)

s_colors=["#00E5FF","#00CFEE","#00BADD","#00FFEE","#00DDE0","#44EEFF"]
s_trails=[ax.plot([],[],color=c,lw=0.4,alpha=0.15,zorder=3)[0] for c in s_colors]
s_dots=[ax.plot([],[],"o",color=c,ms=7,zorder=7,markeredgecolor="white",markeredgewidth=0.7)[0] for c in s_colors]
s_rings=[Circle((0,0),SCAN_R,color=c,fill=False,lw=0.6,ls="--",alpha=0.17,zorder=4) for c in s_colors]
[ax.add_patch(r) for r in s_rings]
s_lbls=[ax.text(0,0,f"S{i+1}",color=s_colors[i],fontsize=4.2,ha="center",fontfamily="monospace",zorder=8) for i in range(6)]
s_prngs=[Circle((0,0),0,color="#FFD166",fill=False,lw=1.4,alpha=0,zorder=6) for _ in scouts]
[ax.add_patch(r) for r in s_prngs]

c_colors=["#FF9F1C","#FF5500","#FFCC00","#FF3399"]
c_dots=[ax.plot([],[],"D",color=c,ms=9,zorder=7,markeredgecolor="white",markeredgewidth=0.8)[0] for c in c_colors]
c_lbls=[ax.text(0,0,f"C{i+1}",color=c_colors[i],fontsize=4.5,ha="center",fontfamily="monospace",zorder=8,fontweight="bold") for i in range(4)]
c_rings=[Circle((0,0),0,color=c,fill=False,lw=1.6,alpha=0,zorder=5) for c in c_colors]
[ax.add_patch(r) for r in c_rings]
c_pths=[ax.plot([],[],"--",color=c,lw=0.6,alpha=0.3,zorder=3)[0] for c in c_colors]

r_colors=["#00FF88","#00DD66","#00BB55","#44FF99","#88FFAA"]
r_dots=[ax.plot([],[],"s",color=c,ms=8,zorder=7,markeredgecolor="white",markeredgewidth=0.6)[0] for c in r_colors]
r_lbls=[ax.text(0,0,f"R{i+1}",color=r_colors[i],fontsize=4.2,ha="center",fontfamily="monospace",zorder=8) for i in range(5)]
r_pths=[ax.plot([],[],"--",color=c,lw=0.5,alpha=0.2,zorder=3)[0] for c in r_colors]

for v in victims:
    col=COLORS.get(v["priority"] or "STABLE")
    v_dots.append(ax.plot([],[],"o",color=col,ms=7,zorder=6,markeredgecolor="white",markeredgewidth=0.5,alpha=0)[0])
    v_lbls.append(ax.text(v["x"],v["y"]+2.8,v["id"],color=col,fontsize=3.8,ha="center",fontfamily="monospace",alpha=0))
    rg=Circle((v["x"],v["y"]),3.0,color=col,fill=False,lw=0.9,alpha=0); ax.add_patch(rg); v_rings.append(rg)

# ── COMMAND PANEL ─────────────────────────────────────────────────────────────
axp.set_facecolor("#030B14"); axp.axis("off")

def hdiv(y, lw=0.8, col="#1B3A5C"):
    axp.plot([0.02,0.98],[y,y],color=col,lw=lw,transform=axp.transAxes)

axp.text(0.5,0.990,"◈  COMMAND PANEL  ◈",color="#FFD166",fontsize=14,fontweight="bold",
         ha="center",va="top",fontfamily="monospace",transform=axp.transAxes)
hdiv(0.974, lw=1.2, col="#FFD166")

sv={}

Y_SUM = 0.960
for i,(k,c) in enumerate([("Detected","#FFFFFF"),("Assessed","#AAFFE0"),("Rescued","#2DC653"),("Aftershocks","#FF6622")]):
    x = 0.03 + i*0.245
    axp.text(x, Y_SUM,   k,   color="#4a6a8a",fontsize=6.5,va="top",fontfamily="monospace",transform=axp.transAxes)
    sv[k]=axp.text(x, Y_SUM-0.035, "—", color=c,fontsize=10.5,va="top",fontfamily="monospace",
                   fontweight="bold",transform=axp.transAxes)

hdiv(0.915, lw=0.8)

Y_RST = 0.900
axp.text(0.03,Y_RST,"ROBOT STATUS",color="#FFD166",fontsize=8.0,fontweight="bold",
         fontfamily="monospace",transform=axp.transAxes)

Y_DR = Y_RST - 0.052
axp.text(0.03,Y_DR,"DRONES",color="#4a8aaa",fontsize=6.5,fontfamily="monospace",transform=axp.transAxes)
sv["DRONE"] =axp.text(0.28,Y_DR,"D1:FLY",color="#AAFFE0",fontsize=7.2,fontweight="bold",
                       fontfamily="monospace",transform=axp.transAxes)
sv["DRONE2"]=axp.text(0.62,Y_DR,"D2:FLY",color="#FFD166",fontsize=7.2,fontweight="bold",
                       fontfamily="monospace",transform=axp.transAxes)

Y_SC = Y_DR - 0.062
axp.text(0.03,Y_SC,"SCOUTS",color="#4a8aaa",fontsize=6.5,fontfamily="monospace",transform=axp.transAxes)
for i in range(6):
    col_x = 0.28 + (i%3)*0.235
    row_y  = Y_SC - 0.024 if i<3 else Y_SC-0.048
    axp.text(col_x,row_y,f"S{i+1}",color=s_colors[i],fontsize=5.8,fontweight="bold",
             fontfamily="monospace",transform=axp.transAxes)
    sv[f"S{i+1}"]=axp.text(col_x,row_y-0.018,"—",color=s_colors[i],fontsize=5.2,
                             fontfamily="monospace",transform=axp.transAxes)

Y_CL = Y_SC - 0.100
axp.text(0.03,Y_CL,"CLEARERS",color="#4a8aaa",fontsize=6.5,fontfamily="monospace",transform=axp.transAxes)
for i in range(4):
    col_x = 0.28 + (i%2)*0.36
    row_y  = Y_CL - 0.026 if i<2 else Y_CL-0.052
    axp.text(col_x,row_y,f"C{i+1}",color=c_colors[i],fontsize=5.8,fontweight="bold",
             fontfamily="monospace",transform=axp.transAxes)
    sv[f"C{i+1}"]=axp.text(col_x,row_y-0.020,"—",color=c_colors[i],fontsize=5.2,
                             fontfamily="monospace",transform=axp.transAxes)

Y_RS = Y_CL - 0.100
axp.text(0.03,Y_RS,"RESCUES",color="#4a8aaa",fontsize=6.5,fontfamily="monospace",transform=axp.transAxes)
for i in range(5):
    col_x = 0.28 + i*0.140
    axp.text(col_x,Y_RS-0.024,f"R{i+1}",color=r_colors[i],fontsize=5.5,fontweight="bold",
             fontfamily="monospace",transform=axp.transAxes)
    sv[f"R{i+1}"]=axp.text(col_x,Y_RS-0.042,"—",color=r_colors[i],fontsize=5.0,
                             fontfamily="monospace",transform=axp.transAxes)

Y_DIV2 = Y_RS - 0.048
hdiv(Y_DIV2, lw=0.8)

Y_VI = Y_DIV2 - 0.028
axp.text(0.03,Y_VI,"VICTIM INTEL",color="#FFD166",fontsize=8.0,fontweight="bold",
         fontfamily="monospace",transform=axp.transAxes)

COL_POSITIONS = [(0.03,"ID"),(0.14,"PRIO"),(0.29,"TEMP"),(0.43,"HR"),
                 (0.56,"SOUND"),(0.74,"SRC"),(0.86,"STATE")]
Y_HDR = Y_VI - 0.030
for cx2,ct in COL_POSITIONS:
    axp.text(cx2,Y_HDR,ct,color="#4a6a8a",fontsize=5.5,fontfamily="monospace",
             fontweight="bold",transform=axp.transAxes)
hdiv(Y_HDR - 0.008, lw=0.6, col="#243a5c")

VROW_H=0.026; vrow_arts=[]
Y_VTOP = Y_HDR - 0.015
for i in range(DETAIL_MAX):
    y=Y_VTOP - i*VROW_H
    row={k:axp.text(x,y,"",color=c,fontsize=5.2,fontfamily="monospace",transform=axp.transAxes,va="top")
         for k,x,c in [("id",0.03,"white"),("prio",0.14,"white"),("temp",0.29,"#aaddaa"),
                        ("hr",0.43,"#aaddaa"),("sound",0.56,"#aaddaa"),("src",0.74,"#AAFFE0"),("state",0.86,"#aaddaa")]}
    vrow_arts.append(row)

Y_DIV3 = Y_VTOP - DETAIL_MAX*VROW_H - 0.002
hdiv(Y_DIV3, lw=0.5, col="#243a5c")

Y_LOG = Y_DIV3 - 0.018
axp.text(0.03,Y_LOG,"EVENT LOG",color="#FFD166",fontsize=8.0,fontweight="bold",
         fontfamily="monospace",transform=axp.transAxes)
LOG_ROWS = 6
lts=[axp.text(0.03,Y_LOG-0.022-i*0.019,"",color="#9ac4d8",fontsize=4.8,
              fontfamily="monospace",transform=axp.transAxes) for i in range(LOG_ROWS)]

LEG_Y = 0.010
for i,(sym,col,desc) in enumerate([("✈","#AAFFE0","Drone(mark)"),("●","#00E5FF","Scouts×6"),
                                    ("◆","#FF9F1C","Clearers×4"),("■","#00FF88","Rescue×5"),
                                    ("⚡","#FF4422","AfterShock")]):
    axp.text(0.02+i*0.196,LEG_Y,f"{sym} {desc}",color=col,fontsize=5.5,
             fontfamily="monospace",transform=axp.transAxes)

# ── DYNAMIC AFTERSHOCK TICK ────────────────────────────────────────────────────
PRIOS=["CRITICAL","MODERATE","STABLE","MODERATE","CRITICAL","STABLE","MODERATE","CRITICAL","STABLE","MODERATE"]
AFTER_HR=[145,115,78,108,152,82,119,47,73,111]

def tick_aftershocks_dynamic():
    for idx,a in enumerate(AFTERSHOCKS):
        if a["spawned"]: continue
        warn_start = a["spawn_frame"] - 120

        if fnum >= warn_start:
            progress = (fnum - warn_start) / 120.0

            a["rumble_phase"] += 0.18
            amp = 0.55 * progress
            rdx = amp * np.sin(a["rumble_phase"]) * a["rumble_dir"][0]
            rdy = amp * np.cos(a["rumble_phase"] * 0.7) * a["rumble_dir"][1]
            a["cx"] = float(a["orig_cx"]) + rdx
            a["cy"] = float(a["orig_cy"]) + rdy

            a["warn_art"].center = (a["cx"], a["cy"])
            pulse = 0.15 + 0.35*np.sin(fnum*0.45)*progress
            a["warn_art"].set_alpha(max(0, pulse))
            a["warn_art"].set_radius(max(a["hw"],a["hh"]) + 3 + 2*progress)

            a["danger_circle"].center = (a["cx"], a["cy"])
            a["danger_circle"].set_alpha(0.06 * progress)

            rebuild_penalty_grid()

            if fnum >= a["spawn_frame"]:
                _do_spawn_aftershock(a, idx)

def _do_spawn_aftershock(a, idx):
    a["cx"], a["cy"] = a["orig_cx"], a["orig_cy"]
    x0,y0=a["cx"]-a["hw"],a["cy"]-a["hh"]; x1,y1=a["cx"]+a["hw"],a["cy"]+a["hh"]
    block(x0,y0,x1,y1); rebuild_nav_grid(); rebuild_penalty_grid()
    a["spawned"]=True
    a["patch"].set_xy((x0,y0)); a["patch"].set_width(x1-x0); a["patch"].set_height(y1-y0)
    a["patch"].set_alpha(0.92)
    a["lbl_art"].set_text(a["label"]); a["lbl_art"].set_position((a["cx"],a["cy"]))
    a["warn_art"].set_alpha(0)
    a["danger_circle"].center=(a["cx"],a["cy"]); a["danger_circle"].set_alpha(0.10)

    log(f"⚡ {a['label']} COLLAPSED ({int(a['cx'])},{int(a['cy'])})")

    vid=f"AV-{idx+1}"; prio=PRIOS[idx]
    fx,fy=nearest_free(int(a["cx"]),int(a["cy"]))
    nv=dict(id=vid,x=float(fx)+0.5,y=float(fy)+0.5,priority=None,thermal=None,hr=None,sound=None,
            true_priority=prio,true_thermal=38.1,true_hr=AFTER_HR[idx],true_sound="Trapped",
            blocked_by=[],ba=a,assessed=False,detected=False,rescued=False,assigned=False,waiting=True)
    victims.append(nv)
    col=COLORS[prio]
    v_dots.append(ax.plot([],[],"o",color=col,ms=7,zorder=6,markeredgecolor="white",markeredgewidth=0.5,alpha=0)[0])
    v_lbls.append(ax.text(float(fx)+0.5,float(fy)+2.8,vid,color=col,fontsize=3.8,ha="center",fontfamily="monospace",alpha=0))
    rg=Circle((float(fx)+0.5,float(fy)+0.5),3.0,color=col,fill=False,lw=0.9,alpha=0); ax.add_patch(rg); v_rings.append(rg)
    a["blocked_victim"]=nv
    log(f"   {vid} [{prio}] HR:{AFTER_HR[idx]}bpm under {a['label']}")
    global_clear_queue.append(("aftershock_obj",a)); dispatch_clearers_now()

# ── ACCESSIBILITY ─────────────────────────────────────────────────────────────
def victim_is_accessible(v):
    a=v.get("ba"); blocked=v.get("blocked_by",[])
    if a and a["label"] not in cleared_aftershocks: return False
    if any(bi not in cleared_rubble for bi in blocked): return False
    return path_exists(BASE,(v["x"],v["y"]))

# ── CLEARER DISPATCH ──────────────────────────────────────────────────────────
def dispatch_clearers_now():
    while global_clear_queue:
        t=global_clear_queue[0]
        if   t[0]=="rubble"         and t[1] in cleared_rubble:               global_clear_queue.pop(0)
        elif t[0]=="aftershock_obj" and t[1]["label"] in cleared_aftershocks: global_clear_queue.pop(0)
        else: break
    for c in clearers:
        if c["status"]!="standby": continue
        task=_pick_next_task_for(c)
        if task: _assign_task_to_clearer(c,task)

def _pick_next_task_for(c):
    for idx,task in enumerate(global_clear_queue):
        if task[0]=="rubble":
            if task[1] in cleared_rubble or ("rubble",task[1]) in global_clear_in_prog: continue
        elif task[0]=="aftershock_obj":
            if task[1]["label"] in cleared_aftershocks or ("aftershock",task[1]["label"]) in global_clear_in_prog: continue
        return global_clear_queue.pop(idx)
    return None

def _assign_task_to_clearer(c,task):
    c["task"]=task; c["status"]="moving"; c["attempt_count"]=0
    if task[0]=="rubble":
        bi=task[1]; x0,y0,x1,y1,lbl=RUBBLE[bi]
        global_clear_in_prog.add(("rubble",bi))
        cx_r,cy_r=(x0+x1)/2,(y0+y1)/2
        approaches=find_approach(cx_r,cy_r,(x1-x0)/2,(y1-y0)/2,offset=3)
        for ap in approaches:
            set_path(c,(ap[0]+0.5,ap[1]+0.5))
            if c["path"]: break
        if not c["path"]:
            fb=nearest_free(int(cx_r),int(cy_r))
            set_path(c,(fb[0]+0.5,fb[1]+0.5))
        log(f"C{c['id']} → {lbl}")
    elif task[0]=="aftershock_obj":
        a=task[1]; global_clear_in_prog.add(("aftershock",a["label"]))
        approaches=find_approach(a["cx"],a["cy"],a["hw"],a["hh"],offset=3)
        for ap in approaches:
            set_path(c,(ap[0]+0.5,ap[1]+0.5))
            if c["path"]: break
        if not c["path"]:
            fb=nearest_free(int(a["cx"]),int(a["cy"]))
            set_path(c,(fb[0]+0.5,fb[1]+0.5))
        log(f"C{c['id']} → {a['label']}")

# ── DRONE DETECTION ───────────────────────────────────────────────────────────
def _queue_victim_for_scout(v):
    if v.get("assessed") or v["rescued"]: return
    if v["id"] in scout_assess_assigned:
        for s in scouts:
            if s.get("assess_target") and s["assess_target"]["id"] == v["id"]:
                return
        scout_assess_assigned.discard(v["id"])
    if v["id"] not in scout_assess_assigned:
        drone_marked_queue.append(v)
        log(f"✈ MARKED {v['id']} → scout queued")


def on_drone_detected(v):
    if v["detected"]: return
    v["_source"]="DRONE"
    if v["id"] not in [d["id"] for d in detected_log]: detected_log.append(v)
    v["detected"]=True
    a=v.get("ba"); blocked=v.get("blocked_by",[])
    if a:
        v["waiting"]=True
        if a["spawned"] and a["label"] not in cleared_aftershocks:
            key=("aftershock_obj",a)
            if not any(t[0]=="aftershock_obj" and t[1]["label"]==a["label"] for t in global_clear_queue) \
               and ("aftershock",a["label"]) not in global_clear_in_prog:
                global_clear_queue.append(key)
            dispatch_clearers_now()
        return
    if blocked:
        v["waiting"]=True
        for bi in blocked:
            if bi not in cleared_rubble \
               and ("rubble",bi) not in global_clear_in_prog \
               and not any(t[0]=="rubble" and t[1]==bi for t in global_clear_queue):
                global_clear_queue.append(("rubble",bi))
        dispatch_clearers_now()
        return
    if victim_is_accessible(v):
        _queue_victim_for_scout(v)

    else:
        log(f"✈ MARKED {v['id']} → no path yet")


def on_scout_assessed(v, scout_id):
    if v.get("assessed"): return
    if not victim_is_accessible(v):
        log(f"{scout_id}: {v['id']} blocked — defer")
        scout_assess_assigned.discard(v["id"])
        return
    v["assessed"]=True
    v["priority"]=v["true_priority"]; v["thermal"]=v["true_thermal"]
    v["hr"]=v["true_hr"]; v["sound"]=v["true_sound"]
    log(f"{scout_id} ASSESSED {v['id']} [{v['priority']}] HR:{v['hr']}")
    on_detected_for_rescue(v)


def on_detected_for_rescue(v):
    if v["assigned"] or v["rescued"]: return
    if not victim_is_accessible(v): v["waiting"]=True; return
    v["assigned"]=True; v["waiting"]=False
    global_rescue_queue.append(v); global_rescue_queue.sort(key=prio_key)
    log(f">> RESCUE: {v['id']} [{v['priority']}]")

# ── SCOUT DISPATCH ────────────────────────────────────────────────────────────
def dispatch_scouts_to_marked():
    for s in scouts:
        if s["status"] not in ("scanning","done","returning") or s.get("assess_target"): continue
        if not drone_marked_queue: break
        best_v=None; best_d=1e9
        for v in list(drone_marked_queue):
            if v["id"] in scout_assess_assigned:
                drone_marked_queue.remove(v); continue
            if not victim_is_accessible(v): continue
            d=np.hypot(s["x"]-v["x"],s["y"]-v["y"])
            if d<best_d: best_d=d; best_v=v
        if not best_v: break
        drone_marked_queue.remove(best_v); scout_assess_assigned.add(best_v["id"])
        s["assess_target"]=best_v; s["status"]="going_to_assess"
        s["saved_path"]=s["path"]; s["saved_pi"]=s["pi"]
        set_path(s,(best_v["x"],best_v["y"]))
        log(f"{s['id']} → ASSESS {best_v['id']}")

# ── REPLAN AFTER CLEAR ────────────────────────────────────────────────────────
def _replan_after_clear():
    rebuild_penalty_grid()
    for r in rescues:
        if r["status"]=="moving" and r["target"]:
            set_path(r,(r["target"]["x"],r["target"]["y"]))
    for si,s in enumerate(scouts):
        if s["status"]=="going_to_assess" and s.get("assess_target"):
            set_path(s,(s["assess_target"]["x"],s["assess_target"]["y"]))
        elif s["status"] in ("scanning","done","returning"):
            s["path"]=build_scout_coverage_path(si); s["pi"]=0
    for c2 in clearers:
        if c2["status"]=="moving" and c2["task"]:
            task=c2["task"]
            if task[0]=="rubble":
                bi=task[1]; x0,y0,x1,y1,_=RUBBLE[bi]; cx_r,cy_r=(x0+x1)/2,(y0+y1)/2
                approaches=find_approach(cx_r,cy_r,(x1-x0)/2,(y1-y0)/2,offset=3)
                for ap in approaches:
                    set_path(c2,(ap[0]+0.5,ap[1]+0.5))
                    if c2["path"]: break
            elif task[0]=="aftershock_obj":
                a=task[1]
                approaches=find_approach(a["cx"],a["cy"],a["hw"],a["hh"],offset=3)
                for ap in approaches:
                    set_path(c2,(ap[0]+0.5,ap[1]+0.5))
                    if c2["path"]: break
    for v in victims:
        if v["rescued"]: continue
        if v.get("assessed") and not v["assigned"] and not v["rescued"]:
            if victim_is_accessible(v):
                v["waiting"]=False; on_detected_for_rescue(v)
            continue
        if v["detected"] and not v.get("assessed"):
            if victim_is_accessible(v):
                _queue_victim_for_scout(v)
        elif not v["detected"]:
            has_blocker=v.get("blocked_by") or v.get("ba")
            if has_blocker and victim_is_accessible(v):
                v["detected"]=True; v["_source"]="CLEAR"
                if v["id"] not in [d["id"] for d in detected_log]: detected_log.append(v)
                _queue_victim_for_scout(v)
                log(f">> {v['id']} unblocked → scout")

# ── LOGIC TICK ────────────────────────────────────────────────────────────────
def tick():
    global fnum; fnum+=1
    tick_aftershocks_dynamic()
    tick_drone(); tick_drone2()
    dispatch_scouts_to_marked()
    for s in scouts: tick_scout(s)
    tick_all_clearers()
    for r in rescues: tick_rescue(r)

    for v in victims:
        if v["rescued"] or v["assigned"]: continue
        if v.get("assessed") and v["waiting"]:
            if victim_is_accessible(v): v["waiting"]=False; on_detected_for_rescue(v)
            continue
        has_blocker=v.get("blocked_by") or v.get("ba")
        if not has_blocker: continue
        if not victim_is_accessible(v): continue
        if not v["detected"]:
            v["detected"]=True; v["_source"]="CLEAR"
            if v["id"] not in [d["id"] for d in detected_log]: detected_log.append(v)
        if not v.get("assessed") and v["id"] not in scout_assess_assigned:
            _queue_victim_for_scout(v)

    global_rescue_queue.sort(key=prio_key)

    if fnum%15==0:
        for r in rescues:
            if r["status"]=="moving" and r["target"] and r["path"]:
                ahead=r["path"][r["pi"]:r["pi"]+5]
                if any(nav_grid[int(np.clip(py,0,G-1)),int(np.clip(px,0,G-1))]==1 for px,py in ahead):
                    set_path(r,(r["target"]["x"],r["target"]["y"]))
        for s in scouts:
            if s["status"]=="going_to_assess" and s.get("assess_target") and s["path"]:
                ahead=s["path"][s["pi"]:s["pi"]+5]
                if any(nav_grid[int(np.clip(py,0,G-1)),int(np.clip(px,0,G-1))]==1 for px,py in ahead):
                    set_path(s,(s["assess_target"]["x"],s["assess_target"]["y"]))
        for c in clearers:
            if c["status"]=="moving" and c["path"] and c["pi"]<len(c["path"]):
                ahead=c["path"][c["pi"]:c["pi"]+5]
                if any(nav_grid[int(np.clip(py,0,G-1)),int(np.clip(px,0,G-1))]==1 for px,py in ahead):
                    if c["task"]:
                        task=c["task"]
                        if task[0]=="rubble":
                            bi=task[1]; x0,y0,x1,y1,_=RUBBLE[bi]; cx_r,cy_r=(x0+x1)/2,(y0+y1)/2
                            approaches=find_approach(cx_r,cy_r,(x1-x0)/2,(y1-y0)/2,offset=3)
                            for ap in approaches:
                                set_path(c,(ap[0]+0.5,ap[1]+0.5))
                                if c["path"]: break
                        elif task[0]=="aftershock_obj":
                            a=task[1]
                            approaches=find_approach(a["cx"],a["cy"],a["hw"],a["hh"],offset=3)
                            for ap in approaches:
                                set_path(c,(ap[0]+0.5,ap[1]+0.5))
                                if c["path"]: break

        active_targets = {s["assess_target"]["id"] for s in scouts if s.get("assess_target")}
        queued_ids = {v["id"] for v in drone_marked_queue}
        for v in victims:
            if v["rescued"] or v.get("assessed"): continue
            if not v["detected"]: continue
            if v["id"] in active_targets or v["id"] in queued_ids: continue
            if victim_is_accessible(v):
                scout_assess_assigned.discard(v["id"])
                _queue_victim_for_scout(v)
                log(f"♻ REQUEUE {v['id']} (limbo)")

    if fnum%60==0:
        rebuild_penalty_grid()
        for r in rescues:
            if r["status"]=="moving" and r["target"]:
                set_path(r,(r["target"]["x"],r["target"]["y"]))
        for s in scouts:
            if s["status"]=="going_to_assess" and s.get("assess_target"):
                set_path(s,(s["assess_target"]["x"],s["assess_target"]["y"]))

# ── DRONE TICKS ───────────────────────────────────────────────────────────────
def tick_drone():
    if drone["status"]=="done": return
    done=move_drone(drone); drone["trail"].append((drone["x"],drone["y"]))
    if len(drone["trail"])>800: drone["trail"].pop(0)
    if drone["scan_flash"]>0: drone["scan_flash"]-=1
    for v in victims:
        if v["detected"]: continue
        if np.hypot(drone["x"]-v["x"],drone["y"]-v["y"])<=DRONE_R:
            drone["scan_flash"]=18; drone["last_found"]=v; on_drone_detected(v)
    if done: drone["status"]="done"; log("✈ D1 COMPLETE")

def tick_drone2():
    if drone2["status"]=="done": return
    done=move_drone(drone2); drone2["trail"].append((drone2["x"],drone2["y"]))
    if len(drone2["trail"])>800: drone2["trail"].pop(0)
    if drone2["scan_flash"]>0: drone2["scan_flash"]-=1
    for v in victims:
        if v["detected"]: continue
        if np.hypot(drone2["x"]-v["x"],drone2["y"]-v["y"])<=DRONE_R:
            drone2["scan_flash"]=18; drone2["last_found"]=v; on_drone_detected(v)
    if done: drone2["status"]="done"; log("✈ D2 COMPLETE")

# ── SCOUT TICK (WITH RETURN TO BASE) ───────────────────────────────────────────
def tick_scout(s):
    # Detect if scout is stuck in place
    if s["last_x"] is not None:
        dist_moved = np.hypot(s["x"]-s["last_x"], s["y"]-s["last_y"])
        if dist_moved < 0.05:  # Not moving
            s["stuck_timer"] += 1
            # If stuck for 5 seconds (150 frames at 30fps), force restart
            if s["stuck_timer"] > 150 and s["status"] in ("scanning", "paused", "assessing"):
                log(f"{s['id']} STUCK (status={s['status']}), forcing restart")
                s["status"] = "scanning"
                s["path"] = build_scout_coverage_path(scouts.index(s))
                s["pi"] = 0
                s["stuck_timer"] = 0
                s["assess_target"] = None
        else:
            s["stuck_timer"] = 0  # Reset if moving
    s["last_x"] = s["x"]
    s["last_y"] = s["y"]
    
    # MISSION END: Force all scouts to return to base when mission complete
    if all_victims_found():
        if s["status"] != "returning" and s["status"] != "done":
            # Force transition to returning
            s["status"]="returning"
            set_path(s, BASE)
            log(f"{s['id']} MISSION COMPLETE → returning")
        elif s["status"] == "done":
            # Already at base, stay done (terminal state)
            return
    
    if s["status"]=="going_to_assess":
        target=s["assess_target"]
        if not target: s["status"]="assessing"; s["pause_t"]=0; return
        if not s["path"] or s["pi"]>=len(s["path"]):
            set_path(s,(target["x"],target["y"]))
            if not s["path"]:
                scout_assess_assigned.discard(target["id"])
                s["assess_target"]=None; s["status"]="assessing"; s["pause_t"]=0; return
        arrived=move(s,SPD["scout"])
        s["trail"].append((s["x"],s["y"]))
        if len(s["trail"])>600: s["trail"].pop(0)
        dist=np.hypot(s["x"]-target["x"],s["y"]-target["y"])
        if arrived or dist<=SCAN_R+0.5:
            on_scout_assessed(target,s["id"])
            s["assess_target"]=None; s["status"]="assessing"; s["pause_t"]=0
        return

    if s["status"]=="assessing":
        s["pause_t"]+=1
        if s["pause_t"]>=PAUSE_F:
            # After assessment, check if all victims found → return to base
            if all_victims_found():
                s["status"]="returning"
                set_path(s, BASE)
                log(f"{s['id']} → BASE (all victims found)")
            else:
                s["status"]="scanning"; s["path"]=s["saved_path"]; s["pi"]=s["saved_pi"]
                if not s["path"]: s["path"]=build_scout_coverage_path(scouts.index(s)); s["pi"]=0
        return

    if s["status"]=="paused":
        s["pause_t"]+=1
        if s["pause_t"]>=PAUSE_F:
            # After pause, check if all victims found → return to base
            if all_victims_found():
                s["status"]="returning"
                set_path(s, BASE)
                log(f"{s['id']} → BASE (all victims found)")
            else:
                s["status"]="scanning"; s["path"]=s["saved_path"]; s["pi"]=s["saved_pi"]
        return

    if s["status"]=="returning":
        # Return to base
        s["stuck_timer"] = 0  # Reset stuck timer while returning
        if not s["path"] or s["pi"]>=len(s["path"]):
            set_path(s, BASE)
        arrived=move(s, SPD["scout"])
        s["trail"].append((s["x"],s["y"]))
        if len(s["trail"])>600: s["trail"].pop(0)
        if arrived:
            s["status"]="done"
            log(f"{s['id']} RETURNED TO BASE ✓ - MISSION END")
        return

    if s["status"] in ("scanning","done"):
        idx=scouts.index(s)
        if not s["path"] or s["pi"]>=len(s["path"]):
            s["path"]=build_scout_coverage_path(idx); s["pi"]=0
            if not s["path"]: return  # Try again next frame, don't give up
        else:
            ahead=s["path"][s["pi"]:s["pi"]+8]
            if any(nav_grid[int(np.clip(py,0,G-1)),int(np.clip(px,0,G-1))]==1 for px,py in ahead):
                s["path"]=build_scout_coverage_path(idx); s["pi"]=0

        arrived=move(s,SPD["scout"])
        s["trail"].append((s["x"],s["y"]))
        if len(s["trail"])>600: s["trail"].pop(0)

        gx,gy=int(s["x"]),int(s["y"]); r=int(SCAN_R)
        for ddx in range(-r,r+1):
            for ddy in range(-r,r+1):
                if ddx*ddx+ddy*ddy<=r*r: scanned.add((gx+ddx,gy+ddy))

        for v in victims:
            if not v["detected"] or v.get("assessed") or v["rescued"]: continue
            if v["id"] in scout_assess_assigned: continue
            if np.hypot(s["x"]-v["x"],s["y"]-v["y"])<=SCAN_R:
                if not victim_is_accessible(v): continue
                scout_assess_assigned.add(v["id"])
                on_scout_assessed(v,s["id"])
                s["saved_path"]=s["path"]; s["saved_pi"]=s["pi"]
                s["status"]="paused"; s["pause_t"]=0

        # Check if all victims found before transitioning to done
        if arrived and s["pi"]>=len(s["path"]):
            if all_victims_found():
                # Mission complete - all victims rescued, return to base
                s["status"]="returning"
                set_path(s, BASE)
                log(f"{s['id']} → BASE (ALL VICTIMS RESCUED - MISSION END)")
            else:
                # Not done yet, restart coverage path
                s["path"]=build_scout_coverage_path(scouts.index(s)); s["pi"]=0
    
    # Monitor "done" state - TERMINAL STATE when mission complete and at base
    if s["status"]=="done":
        # Done state is terminal - scouts stay at base
        # Just idle here, don't restart anything
        return

# ── CLEARER TICK ──────────────────────────────────────────────────────────────
def tick_all_clearers():
    dispatch_clearers_now()
    for c in clearers:
        if c["status"]=="moving":
            # Detect if clearer is stuck in place
            if c["last_x"] is not None:
                dist_moved = np.hypot(c["x"]-c["last_x"], c["y"]-c["last_y"])
                if dist_moved < 0.1:  # Not moving
                    c["stuck_timer"] += 1
                else:
                    c["stuck_timer"] = 0  # Reset if moving
                # If stuck for too long (3 seconds at 30fps), skip task
                if c["stuck_timer"] > 90:
                    task = c["task"]
                    if task:
                        log(f"C{c['id']} STUCK, skipping {'rubble' if task[0]=='rubble' else task[1]['label']}")
                        if task[0]=="rubble":
                            global_clear_in_prog.discard(("rubble",task[1]))
                        else:
                            global_clear_in_prog.discard(("aftershock",task[1]["label"]))
                        c["task"] = None
                        c["status"] = "standby"
                        c["stuck_timer"] = 0
            c["last_x"] = c["x"]
            c["last_y"] = c["y"]
            arrived=move(c,SPD["clearer"])
            if arrived and c["task"]:
                task=c["task"]
                if task[0]=="rubble":
                    bi=task[1]; x0,y0,x1,y1,_=RUBBLE[bi]; cx_r,cy_r=(x0+x1)/2,(y0+y1)/2
                    if np.hypot(c["x"]-cx_r,c["y"]-cy_r)<max(x1-x0,y1-y0)+5:
                        c["status"]="clearing"; c["timer"]=0
                    else:
                        approaches=find_approach(cx_r,cy_r,(x1-x0)/2,(y1-y0)/2,offset=2)
                        for ap in approaches:
                            set_path(c,(ap[0]+0.5,ap[1]+0.5))
                            if c["path"]: break
                elif task[0]=="aftershock_obj":
                    a=task[1]
                    if np.hypot(c["x"]-a["cx"],c["y"]-a["cy"])<max(a["hw"],a["hh"])+6:
                        c["status"]="clearing"; c["timer"]=0
                    else:
                        c["attempt_count"]=c.get("attempt_count",0)+1
                        if c["attempt_count"]>8:
                            # Can't reach this aftershock, reassign to another clearer
                            log(f"C{c['id']} FAILED to reach {a['label']}, reassigning")
                            global_clear_in_prog.discard(("aftershock",a["label"]))
                            global_clear_queue.insert(0,("aftershock_obj",a))
                            c["task"]=None; c["status"]="standby"
                        else:
                            approaches=find_approach(a["cx"],a["cy"],a["hw"],a["hh"],offset=2)
                            for ap in approaches:
                                set_path(c,(ap[0]+0.5,ap[1]+0.5))
                                if c["path"]: break
            elif arrived:
                c["status"]="returning"; set_path(c,BASE); c["stuck_timer"]=0; c["last_x"]=None

        elif c["status"]=="clearing":
            c["timer"]+=1
            if c["timer"]>=FPS*2:
                task=c["task"]
                if task[0]=="rubble":
                    bi=task[1]; x0,y0,x1,y1,_=RUBBLE[bi]
                    grid[max(0,int(y0)):min(G,int(y1)+1),max(0,int(x0)):min(G,int(x1)+1)]=0
                    cleared_rubble.add(bi); global_clear_in_prog.discard(("rubble",bi))
                    rubble_patches[bi].set_alpha(0.04); rubble_patches[bi].set_facecolor("#040100")
                    log(f"C{c['id']}: {RUBBLE[bi][4]} ✓ CLEARED")
                elif task[0]=="aftershock_obj":
                    a=task[1]
                    grid[max(0,int(a["cy"]-a["hh"])):min(G,int(a["cy"]+a["hh"])+1),
                         max(0,int(a["cx"]-a["hw"])):min(G,int(a["cx"]+a["hw"])+1)]=0
                    cleared_aftershocks.add(a["label"])
                    global_clear_in_prog.discard(("aftershock",a["label"]))
                    a["patch"].set_alpha(0.07); a["patch"].set_facecolor("#080100")
                    a["danger_circle"].set_alpha(0)
                    log(f"C{c['id']}: {a['label']} ✓ CLEARED")
                rebuild_nav_grid(); rebuild_penalty_grid(); _replan_after_clear()
                c["task"]=None
                nxt=_pick_next_task_for(c)
                if nxt: _assign_task_to_clearer(c,nxt); log(f"C{c['id']} chaining")
                else:   c["status"]="returning"; set_path(c,BASE); c["stuck_timer"]=0; c["last_x"]=None

        elif c["status"]=="returning":
            if move(c,SPD["clearer"]): c["status"]="standby"

# ── RESCUE TICK ───────────────────────────────────────────────────────────────
def tick_rescue(r):
    if r["status"]=="standby":
        if global_rescue_queue:
            global_rescue_queue.sort(key=prio_key)
            v=global_rescue_queue.pop(0)
            r["target"]=v; r["status"]="moving"; set_path(r,(v["x"],v["y"]))
            log(f"{r['id']} → {v['id']} [{v['priority']}]")
    elif r["status"]=="moving":
        if global_rescue_queue and r["target"]:
            global_rescue_queue.sort(key=prio_key)
            if prio_key(global_rescue_queue[0]) < prio_key(r["target"]):
                global_rescue_queue.append(r["target"]); global_rescue_queue.sort(key=prio_key)
                v=global_rescue_queue.pop(0); r["target"]=v; set_path(r,(v["x"],v["y"]))
                log(f"{r['id']} PREEMPT → {v['id']}")
        if not r["path"] or r["pi"]>=len(r["path"]):
            if r["target"]: set_path(r,(r["target"]["x"],r["target"]["y"]))
        if move(r,SPD["rescue"]): r["status"]="rescuing"; r["timer"]=0
    elif r["status"]=="rescuing":
        r["timer"]+=1
        if r["timer"]>=FPS*2:
            r["target"]["rescued"]=True; log(f"{r['id']}: {r['target']['id']} SAVED ✓")
            r["target"]=None; r["status"]="returning"; set_path(r,BASE)
    elif r["status"]=="returning":
        if move(r,SPD["rescue"]): r["status"]="standby"

# ── DRAW ──────────────────────────────────────────────────────────────────────
def draw(_):
    tick()
    if scanned:
        pts=np.array([[gx+0.5,gy+0.5] for gx,gy in scanned if 0<=gx<G and 0<=gy<G])
        if len(pts): sc_scat.set_offsets(pts)

    for a in AFTERSHOCKS:
        if not a["spawned"]:
            warn_start = a["spawn_frame"] - 120
            if fnum >= warn_start:
                a["warn_art"].center = (a["cx"], a["cy"])
                a["danger_circle"].center = (a["cx"], a["cy"])

    for dr,tr_art,dot,ring,lbl,flash,beam,name,lbl_col,lbl_dy in [
        (drone, drone_trail_art,drone_dot,drone_ring,drone_lbl,drone_flash,drone_beam,"D1","#AAFFE0",3.2),
        (drone2,drone2_trail_art,drone2_dot,drone2_ring,drone2_lbl,drone2_flash,drone2_beam,"D2","#FFD166",-3.2),
    ]:
        if len(dr["trail"])>1: tx,ty=zip(*dr["trail"]); tr_art.set_data(tx,ty)
        dot.set_data([dr["x"]],[dr["y"]]); ring.center=(dr["x"],dr["y"])
        lbl.set_position((dr["x"],dr["y"]+lbl_dy))
        if dr["scan_flash"]>0 and dr["last_found"]:
            prog=dr["scan_flash"]/18; ring.set_alpha(0.12+0.35*prog)
            flash.center=(dr["x"],dr["y"]); flash.set_radius(DRONE_R*prog*0.6); flash.set_alpha(0.18*prog)
            v=dr["last_found"]; beam.set_data([dr["x"],v["x"]],[dr["y"],v["y"]]); beam.set_alpha(0.55*prog)
        else: ring.set_alpha(0.10); flash.set_alpha(0); beam.set_alpha(0)
        if dr["status"]=="done": dot.set_alpha(0.3); lbl.set_text(f"{name}✓"); lbl.set_color("#44AA77")
        else: dot.set_alpha(1.0); lbl.set_text(name); lbl.set_color(lbl_col)

    for i,s in enumerate(scouts):
        if len(s["trail"])>1: tx,ty=zip(*s["trail"]); s_trails[i].set_data(tx,ty)
        s_dots[i].set_data([s["x"]],[s["y"]]); s_rings[i].center=(s["x"],s["y"])
        s_rings[i].set_alpha(0.18 if s["status"] in ("scanning","going_to_assess") else 0)
        s_lbls[i].set_position((s["x"],s["y"]+2.5))
        if s["status"] in ("paused","assessing"):
            prog=s["pause_t"]/PAUSE_F
            s_prngs[i].center=(s["x"],s["y"]); s_prngs[i].set_radius(2+6*prog); s_prngs[i].set_alpha(0.7-0.65*prog)
            s_dots[i].set_markerfacecolor("#00FF99")
        elif s["status"]=="going_to_assess":
            s_dots[i].set_markerfacecolor("#AAFFE0"); s_prngs[i].set_alpha(0)
        elif s["status"]=="returning":
            s_dots[i].set_markerfacecolor("#FFB347"); s_prngs[i].set_alpha(0)
        else:
            s_prngs[i].set_alpha(0); s_dots[i].set_markerfacecolor(s_colors[i])

    for i,c in enumerate(clearers):
        c_dots[i].set_data([c["x"]],[c["y"]]); c_lbls[i].set_position((c["x"],c["y"]+2.5))
        c_dots[i].set_ms(14 if c["status"]=="clearing" else 9)
        if c["status"]=="clearing":
            prog=c["timer"]/(FPS*2); c_rings[i].center=(c["x"],c["y"])
            c_rings[i].set_radius(2+5*prog); c_rings[i].set_alpha(0.5-0.4*prog)
        else: c_rings[i].set_alpha(0)
        if c["path"] and c["pi"]<len(c["path"]):
            rem=c["path"][c["pi"]:]
            if rem: px,py=zip(*rem); c_pths[i].set_data(px,py)
        else: c_pths[i].set_data([],[])

    for i,r in enumerate(rescues):
        r_dots[i].set_data([r["x"]],[r["y"]]); r_lbls[i].set_position((r["x"],r["y"]+2.5))
        r_dots[i].set_ms(11 if r["status"]=="rescuing" else 8)
        if r["path"] and r["pi"]<len(r["path"]):
            rem=r["path"][r["pi"]:]
            if rem: px,py=zip(*rem); r_pths[i].set_data(px,py)
        else: r_pths[i].set_data([],[])

    for v,dot,lbl,ring in zip(victims,v_dots,v_lbls,v_rings):
        if v["detected"]:
            dot.set_data([v["x"]],[v["y"]]); dot.set_alpha(1); lbl.set_alpha(1)
            if not v.get("assessed"):
                dot.set_color("#888888"); lbl.set_color("#888888")
                ring.set_edgecolor("#888888"); ring.set_alpha(0.12+0.08*np.sin(fnum*0.5)); ring.set_visible(True)
            elif v["rescued"]:
                dot.set_color("#1c3820"); ring.set_visible(False); lbl.set_color("#2a5030")
            else:
                col=COLORS[v["priority"]]; dot.set_color(col); lbl.set_color(col)
                ring.set_edgecolor("#FF9F1C" if v["waiting"] else col)
                ring.set_alpha(0.25+0.14*np.sin(fnum*0.28)); ring.set_visible(True)

    det=sum(1 for v in victims if v["detected"])
    assd=sum(1 for v in victims if v.get("assessed"))
    resc=sum(1 for v in victims if v["rescued"])
    asn=sum(1 for a in AFTERSHOCKS if a["spawned"]); acl=len(cleared_aftershocks)
    sv["Detected"].set_text(f"{det}/{len(victims)}")
    sv["Assessed"].set_text(f"{assd}/{len(victims)}")
    sv["Rescued"].set_text(f"{resc}/{len(victims)}")
    sv["Aftershocks"].set_text(f"{asn}↯/{acl}✓")
    sv["DRONE"].set_text("D1:DONE✓" if drone["status"]=="done" else f"D1:FLY {drone['wi']}/{len(drone['wps'])}")
    sv["DRONE2"].set_text("D2:DONE✓" if drone2["status"]=="done" else f"D2:FLY {drone2['wi']}/{len(drone2['wps'])}")

    for i,s in enumerate(scouts):
        at=s.get("assess_target"); st=s["status"]
        if at:       sv[f"S{i+1}"].set_text(f"→{at['id']}")
        elif st in ("paused","assessing"): sv[f"S{i+1}"].set_text("ASSESS")
        elif st == "returning": sv[f"S{i+1}"].set_text("RTN-BASE")
        elif s["pi"]<len(s["path"]): sv[f"S{i+1}"].set_text("SCAN")
        else:        sv[f"S{i+1}"].set_text("DONE✓")

    for i,c in enumerate(clearers):
        task=c["task"]
        if task and c["status"] in ("moving","clearing"):
            lbl2=RUBBLE[task[1]][4] if task[0]=="rubble" else task[1]["label"]
            sv[f"C{i+1}"].set_text(f"{c['status'][:3].upper()} {lbl2[:8]}")
        else: sv[f"C{i+1}"].set_text(c["status"].upper()[:8])

    for i,r in enumerate(rescues):
        sv[f"R{i+1}"].set_text(r["status"][:4].upper()+(f"→{r['target']['id']}" if r["target"] else ""))

    recent=sorted([v for v in detected_log if v.get("assessed")],key=lambda v:(prio_key(v),v["id"]))[:DETAIL_MAX]
    unassessed=[v for v in detected_log if not v.get("assessed")][:max(0,DETAIL_MAX-len(recent))]
    display=recent+unassessed
    for i,row in enumerate(vrow_arts):
        if i<len(display):
            v=display[i]; a=v.get("ba")
            if v["rescued"]: state_str,state_col="SAVED","#2DC653"
            elif not v.get("assessed"): state_str,state_col="UNKN","#888888"
            elif a and a["label"] not in cleared_aftershocks: state_str,state_col="BLKD","#FF3300"
            elif v["waiting"]: state_str,state_col="WAIT","#FF9F1C"
            elif v["assigned"]: state_str,state_col="RESC","#00E5FF"
            else: state_str,state_col="FREE","#aaddaa"
            hr_val=v.get("hr")
            hr_str="—" if hr_val is None else (f"{hr_val}!" if (hr_val<50 or hr_val>140) else (f"{hr_val}^" if hr_val>=100 else str(hr_val)))
            hr_col="#aaddaa" if hr_val is None else ("#E63946" if (hr_val<50 or hr_val>140) else ("#F4A261" if hr_val>=100 else "#2DC653"))
            row["id"].set_text(v["id"]); row["id"].set_color("#2a5030" if v["rescued"] else "white")
            row["prio"].set_text(v["priority"][:3] if v["priority"] else "???")
            row["prio"].set_color(COLORS.get(v["priority"],"#888888"))
            row["temp"].set_text(f"{v['thermal']}°" if v["thermal"] else "—")
            row["hr"].set_text(hr_str); row["hr"].set_color(hr_col)
            row["sound"].set_text((v["sound"] or "—")[:5])
            src=v.get("_source","?"); row["src"].set_text(src[:5])
            row["src"].set_color("#AAFFE0" if src=="DRONE" else "#00E5FF")
            row["state"].set_text(state_str); row["state"].set_color(state_col)
        else:
            for k in ["id","prio","temp","hr","sound","src","state"]: row[k].set_text("")

    for i,lt in enumerate(lts):
        idx=i-(LOG_ROWS-len(logs)); lt.set_text(logs[idx] if idx>=0 else "")

ani=FuncAnimation(fig,draw,frames=20000,interval=1000//FPS,blit=False,repeat=False)
plt.tight_layout(pad=0.4)
plt.show()
