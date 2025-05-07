import argparse
import random
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('API_KEY'))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# -----------------------------
# Global Defaults (will be overridden)
# -----------------------------
MIN_MAIN_STEPS = MAX_MAIN_STEPS = SIDE_QUEST_CHANCE = DECISION_FORK_CHANCE = None
MAX_TASKS_PER_NODE = 2

# Constants
QDS_WEIGHTS = {
    'pokemon': 0.15,      # Number of Pokémon
    'steps': 0.10,        # Number of quest steps
    'battles': 0.10,      # Sum of enemy levels
    'conditions': 0.05,   # Special conditions (e.g., optional steps)
    'catch_rate': 0.40,   # 1 - (catch_rate / 255)
    'base_stats': 0.20    # Normalized BST (Base Stat Total)
}

# Min/Max BST for normalization (adjust based on your game's Pokémon)
MIN_BST = 200    # e.g., Magikarp
MAX_BST = 680    # e.g., Mewtwo

def configure_difficulty(difficulty: int):
    global MIN_MAIN_STEPS, MAX_MAIN_STEPS, SIDE_QUEST_CHANCE, DECISION_FORK_CHANCE, MAX_TASKS_PER_NODE
    if difficulty == 1:
        MIN_MAIN_STEPS, MAX_MAIN_STEPS = 2, 3
        SIDE_QUEST_CHANCE, DECISION_FORK_CHANCE = 0.4, 0.1
        MAX_TASKS_PER_NODE = 1
    elif difficulty == 2:
        MIN_MAIN_STEPS, MAX_MAIN_STEPS = 3, 5
        SIDE_QUEST_CHANCE, DECISION_FORK_CHANCE = 0.6, 0.3
        MAX_TASKS_PER_NODE = 2
    else:
        MIN_MAIN_STEPS, MAX_MAIN_STEPS = 4, 6
        SIDE_QUEST_CHANCE, DECISION_FORK_CHANCE = 0.8, 0.5
        MAX_TASKS_PER_NODE = 3

# -----------------------------
# Quest Tile Definitions & Adjacencies
# -----------------------------
quest_tiles = {
    "starters": [
        {"id": "talkToHumanNPC",       "weight": 1.0},
        {"id": "talkToPokemonNPC",     "weight": 0.8},
        {"id": "findClueItem",         "weight": 0.7},
        {"id": "overhearConversation", "weight": 0.5},
        {"id": "readSignOrNote",       "weight": 0.6},
    ],
    "steps": [
        {"id": "collectXItems",     "weight": 1.0},
        {"id": "battleXPokemon",    "weight": 1.0},
        {"id": "battleXTrainers",   "weight": 0.9},
        {"id": "encounterXPokemon", "weight": 0.8},
        {"id": "findXPokemon",      "weight": 1.0},
        {"id": "followXPokemon",    "weight": 0.6},
        {"id": "deliverXItem",      "weight": 0.9},
        {"id": "escortXNPC",        "weight": 0.5},
        {"id": "useHMAbility",      "weight": 0.7},
    ],
    "completions": [
        {"id": "receivePokemon", "weight": 0.5},
        {"id": "receiveMoney",   "weight": 1.0},
        {"id": "receiveItem",    "weight": 0.8},
        {"id": "returnToNPC",    "weight": 1.0},
    ]
}

starter_adjacency = {
    "talkToHumanNPC":       ["findClueItem", "readSignOrNote"],
    "talkToPokemonNPC":     ["overhearConversation", "findClueItem"],
    "findClueItem":         ["readSignOrNote"],
    "overhearConversation": ["readSignOrNote"],
    "readSignOrNote":       []
}

step_adjacency = {
    "collectXItems":     ["deliverXItem", "useHMAbility"],
    "battleXPokemon":    ["encounterXPokemon", "battleXTrainers"],
    "battleXTrainers":   ["battleXPokemon"],
    "encounterXPokemon": ["battleXPokemon", "followXPokemon"],
    "findXPokemon":      ["followXPokemon"],
    "followXPokemon":    ["escortXNPC"],
    "deliverXItem":      ["useHMAbility"],
    "escortXNPC":        ["useHMAbility"],
    "useHMAbility":      []
}

completion_adjacency = {
    "receiveItem":    [],
    "receivePokemon": [],
    "receiveMoney":   [],
    "returnToNPC":    ["receiveItem", "receivePokemon", "receiveMoney"]
}

class Node:
    def __init__(self, id, type_, optional=False):
        self.id = id
        self.type = type_
        self.optional = optional
        self.children = []
        self.actions = []

class QuestGraph:
    def __init__(self):
        self.nodes = {}
        self.start = None

    def add_node(self, id, type_, optional=False):
        node = Node(id, type_, optional)
        self.nodes[id] = node
        if self.start is None:
            self.start = node
        return node

    def add_edge(self, frm, to):
        self.nodes[frm].children.append(self.nodes[to])

def generate_task_chain(start, max_len, adj):
    chain, cur = [start], start
    while len(chain) < max_len:
        opts = adj.get(cur, [])
        if not opts:
            break
        cur = random.choice(opts)
        chain.append(cur)
    return chain

def assign_tile_actions(node):
    if node.type == "starter":
        pool, adj, L = quest_tiles["starters"], starter_adjacency, MAX_TASKS_PER_NODE
    elif node.type == "step":
        pool, adj, L = quest_tiles["steps"], step_adjacency, MAX_TASKS_PER_NODE
    elif node.type in ("completion", "decision"):
        pool, adj, L = quest_tiles["completions"], completion_adjacency, 1
    else:
        return

    choice = random.choices(
        [t["id"] for t in pool],
        weights=[t["weight"] for t in pool]
    )[0]
    node.actions = generate_task_chain(choice, L, adj)

def populate_quest(graph: QuestGraph):
    for node in graph.nodes.values():
        assign_tile_actions(node)

def generate_dynamic_quest_structure():
    q = QuestGraph()
    prev = q.add_node("start", "starter")
    main_steps = []

    for i in range(random.randint(MIN_MAIN_STEPS, MAX_MAIN_STEPS)):
        step = q.add_node(f"main{i+1}", "step")
        q.add_edge(prev.id, step.id)
        main_steps.append(step)
        prev = step

    if random.random() < SIDE_QUEST_CHANCE:
        base = random.choice(main_steps)
        s1 = q.add_node("side1", "step", optional=True)
        q.add_edge(base.id, "side1")
        if random.random() < 0.5:
            s2 = q.add_node("side2", "step", optional=True)
            q.add_edge("side1", "side2")
            prev_side = s2
        else:
            prev_side = s1
        end = q.add_node("sideComplete", "completion", optional=True)
        q.add_edge(prev_side.id, "sideComplete")

    if random.random() < DECISION_FORK_CHANCE:
        dec = q.add_node("mainDecision", "step")
        q.add_edge(prev.id, "mainDecision")
        for c in ("decisionA", "decisionB"):
            q.add_node(c, "completion")
            q.add_edge("mainDecision", c)
    else:
        q.add_node("complete", "completion")
        q.add_edge(prev.id, "complete")

    return q

# -----------------------------
# CSV Parsing into Structures
# -----------------------------
def load_route_data(enc_csv, tr_csv):
    enc = pd.read_csv(enc_csv)
    tr  = pd.read_csv(tr_csv)

    # Wild Pokémon (Day)
    day = enc[enc["Time"] == "Day"]
    wild = []
    for _, r in day.iterrows():
        # normalize catch rate: if it’s a string with “%”, strip it;
        # otherwise assume it’s the classic 0–255 catch value and divide by 255.
        cr = r["Catch Rate"]
        if isinstance(cr, str) and cr.endswith("%"):
            catch_rate = float(cr.strip("%")) / 100
        else:
            catch_rate = float(cr) / 255

        # parse raw catch (0–255) and normalized fraction
        raw_cr = int(r["Catch Rate"])
        frac_cr = raw_cr / 255.0
        wild.append({
            "name":         r["Pokemon"],
            "level_range":  r["Level(s)"],
            "enc_rate":     float(r["Encounter Rate"].strip("%")) / 100,
            "catch_rate":   frac_cr,
            "catch_raw":    raw_cr,
            "base_stat":    int(r["Total Base Stats"])
        })

    # Trainers grouped
    grp = tr.groupby("Trainer Name").agg({
        "Pokemon":  lambda x: ", ".join(x),
        "Level(s)": lambda x: ", ".join(str(i) for i in x)
    }).reset_index()

    trainers = []
    for _, r in grp.iterrows():
        pkmns = [p.strip() for p in r["Pokemon"].split(",")]
        lvls  = [int(s) for s in r["Level(s)"].split(",")]
        trainers.append({
            "name":     r["Trainer Name"],
            "pokemons": pkmns,
            "levels":   lvls,
            "strength": sum(lvls)
        })

    return wild, trainers

# -----------------------------
# Difficulty‐based Selectors
# -----------------------------
def select_wild(wild_list, diff):
    """
    Pick one wild Pokémon weighted by a difficulty metric that combines:
      • Rarity   = 1 - enc_rate
      • Catch-hardness = 1 - catch_rate
      • Strength = base_stat / max_base_stat

    Easy (diff=1)   → favors low-difficulty Pokémon by inverting the metric
    Medium (diff=2) → uniform random
    Hard (diff=3)   → favors high-difficulty Pokémon (higher metric)
    """
    # find max base stats for normalization
    max_base = max(w["base_stat"] for w in wild_list) or 1

    weights = []
    for w in wild_list:
        enc_score   = 1.0 - w["enc_rate"]
        catch_score = 1.0 - w["catch_rate"]
        base_score  = w["base_stat"] / max_base
        metric = enc_score + catch_score + base_score

        if diff == 1:
            # easy: pick low-metric Pokémon
            weight = 1.0 / (metric + 0.01)
        elif diff == 2:
            # medium: all equal
            weight = 1.0
        else:
            # hard: pick high-metric Pokémon
            weight = metric

        weights.append(weight)

    return random.choices(wild_list, weights=weights, k=1)[0]

def select_trainer(tr_list, diff):
    if diff == 1:
        pool = [t for t in tr_list if t["strength"] <= 15]
    elif diff == 2:
        pool = [t for t in tr_list if 15 < t["strength"] <= 30]
    else:
        pool = [t for t in tr_list if t["strength"] > 30]
    return random.choice(pool or tr_list)

def instantiate_action(action_id, diff, wild_list, tr_list):
    if action_id.startswith("battleXPokemon"):
        w = select_wild(wild_list, diff)
        return f"battle wild {w['name']} (Lv {w['level_range']})"
    if action_id.startswith("encounterXPokemon") or action_id.startswith("findXPokemon"):
        w = select_wild(wild_list, diff)
        verb = "encounter" if "encounter" in action_id else "find"
        return f"{verb} wild {w['name']} (Lv {w['level_range']})"
    if action_id.startswith("battleXTrainers"):
        t = select_trainer(tr_list, diff)
        team = ", ".join(f"{p} (Lv {lv})" for p, lv in zip(t["pokemons"], t["levels"]))
        return f"battle Trainer {t['name']} with {team}"
    if action_id.startswith("collectX Items") or action_id.startswith("deliverXItem"):
        item = random.choice(["Potion", "Poké Ball", "Super Potion", "Antidote"])
        verb = "collect" if "collect" in action_id else "deliver"
        return f"{verb} {item}"
    return action_id

def instantiate_quest_actions(graph, diff, wild_list, tr_list):
    for node in graph.nodes.values():
        node.actions = [
            instantiate_action(a, diff, wild_list, tr_list)
            for a in node.actions
        ]

# -----------------------------
# Quest Difficulty Score
# -----------------------------
def calculate_qds(quest_data, difficulty):
    """
    Calculate QDS using original weights and light difficulty scaling at the end.
    This keeps natural structure-based scoring, but with consistent upward bias.
    """

    # Light difficulty scaling (final boost only)
    difficulty_boost = {
        1: 0.9,
        2: 1.0,
        3: 1.15  # NOT 1.3 — that was too much
    }[difficulty]

    # Normalize catch rate
    if not quest_data['catch_rate']:
        avg_catch_rate = 0
    else:
        catch_rates = [1 - (rate / 255) for rate in quest_data['catch_rate']]
        avg_catch_rate = sum(catch_rates) / len(catch_rates)

    # Normalize base stats
    if not quest_data['base_stats']:
        avg_base_stats = 0
    else:
        base_stats_normalized = [
            (bst - MIN_BST) / (MAX_BST - MIN_BST)
            for bst in quest_data['base_stats']
        ]
        avg_base_stats = sum(base_stats_normalized) / len(base_stats_normalized)

    # Original weights applied to normalized components
    components = {
        'pokemon': QDS_WEIGHTS['pokemon'] * min(quest_data['pokemon'], 10) / 10,
        'steps': QDS_WEIGHTS['steps'] * min(quest_data['steps'], 10) / 10,
        'battles': QDS_WEIGHTS['battles'] * min(quest_data['battles'], 50) / 50,
        'conditions': QDS_WEIGHTS['conditions'] * min(quest_data['conditions'], 5) / 5,
        'catch_rate': QDS_WEIGHTS['catch_rate'] * avg_catch_rate,
        'base_stats': QDS_WEIGHTS['base_stats'] * avg_base_stats,
    }

    # Final QDS with difficulty bias
    raw_score = sum(components.values())
    qds = min(raw_score * 100 * difficulty_boost, 100)

    return {
        'qds': round(qds, 1),
        'components': components
    }



def extract_quest_summary(graph, difficulty, wild_list, tr_list):
    starter = graph.start.actions[0] if graph.start.actions else ""
    steps = [n.actions[0] for n in graph.nodes.values() if n.type == "step"]
    comp = [n.actions[0] for n in graph.nodes.values() if n.type == "completion"]

    # Track only used Pokémon and trainers
    used_pokemon_names = set()
    used_trainer_names = set()
    for node in graph.nodes.values():
        for action in node.actions:
            if "wild" in action:
                parts = action.split()
                if len(parts) >= 3:
                    used_pokemon_names.add(parts[2])
            elif "Trainer" in action:
                parts = action.split()
                if "Trainer" in parts:
                    idx = parts.index("Trainer")
                    if idx + 1 < len(parts):
                        used_trainer_names.add(parts[idx + 1])

    used_wilds = [w for w in wild_list if w["name"] in used_pokemon_names]
    used_trainers = [t for t in tr_list if t["name"] in used_trainer_names]

    qds_input = {
        'pokemon': len(used_wilds) + sum(len(t['pokemons']) for t in used_trainers),
        'steps': len(steps),
        'battles': sum(sum(t['levels']) for t in used_trainers),
        'conditions': len([n for n in graph.nodes.values() if n.optional]),
        'catch_rate': [w['catch_raw'] for w in used_wilds],
        'base_stats': (
            [w['base_stat'] for w in used_wilds] +
            [w['base_stat'] for t in used_trainers for w in used_wilds if w['name'] in t['pokemons']]
        ),
    }

    qds_data = calculate_qds(qds_input, difficulty)
    avg_catch_difficulty = qds_data['components']['catch_rate'] / QDS_WEIGHTS['catch_rate']

    return {
        "starter": starter,
        "steps": steps,
        "completion": comp[0] if comp else "",
        "qds": qds_data['qds'],  # ✅ no double-scaling
        "qds_components": {
            'pokemon_count': qds_input['pokemon'],
            'steps': len(steps),
            'battle_strength': qds_input['battles'],
            'special_conditions': qds_input['conditions'],
            'avg_catch_difficulty': round(avg_catch_difficulty * 100, 1),
            'avg_base_stats': round((qds_data['components']['base_stats'] / QDS_WEIGHTS['base_stats']) * 100, 1)
        }
    }


# -----------------------------
# Printing & Extraction
# -----------------------------
def print_actions(node, wild_list, visited=None, prefix="", is_last=True):
    """
    Recursively prints the quest graph, and for any wild-Pokémon actions,
    appends (Lv range; Catch Rate X%; Base Stats Y).
    """
    if visited is None:
        visited = set()
    if node.id in visited:
        return
    visited.add(node.id)

    connector = "└── " if is_last else "├── "
    opt = " (optional)" if node.optional else ""
    print(prefix + connector + node.id + opt)

    child_prefix = prefix + ("    " if is_last else "│   ")
    for action in node.actions:
        # If this is a wild Pokemon action, augment with catch rate & base stat
        if any(action.startswith(p) for p in ("battle wild", "encounter wild", "find wild")):
            name = action.split()[2]
            entry = next((w for w in wild_list if w["name"] == name), None)
            if entry:
                detail = (f"(Lv {entry['level_range']}; "
                          f"Catch {entry['catch_raw']}; "
                          f"Base Stats {entry['base_stat']})")
                print(child_prefix + f"• {action} {detail}")
            else:
                print(child_prefix + f"• {action}")
        else:
            print(child_prefix + f"• {action}")

    for i, ch in enumerate(node.children):
        print_actions(ch, wild_list, visited, child_prefix, i == len(node.children) - 1)

def extract_main_nodes(graph):
    main, cur, vis = [], graph.start, set()
    while cur and cur.id not in vis:
        vis.add(cur.id)
        if cur.type == "step":
            main.append(cur)
        children = [c for c in cur.children if not c.optional]
        cur = children[0] if children else None
    return main

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--difficulty",
        type=int, choices=[1,2,3],
        default=1,
        help="1=easy,2=medium,3=hard"
    )
    parser.add_argument(
        "--encounters",
        default="route_29_encounters.csv"
    )
    parser.add_argument(
        "--trainers",
        default="route_29_trainers.csv"
    )
    args = parser.parse_args()

    configure_difficulty(args.difficulty)
    wild_list, tr_list = load_route_data(args.encounters, args.trainers)

    wild_summary = ", ".join(
        f"{w['name']} (Lv {w['level_range']}, {int(w['enc_rate']*100)}%)"
        for w in wild_list
    )
    trainer_info = "\n".join(
        f"{t['name']}: {', '.join(p for p in t['pokemons'])} "
        f"(Lv {', '.join(map(str,t['levels']))})"
        for t in tr_list[:3]
    )

    output_path = "generated_quest_script.txt"
    with open(output_path, "w") as f:

        for i in range(3):
            print(f"\n=== Generated Quest #{i+1} ===")
            quest = generate_dynamic_quest_structure()
            populate_quest(quest)
            instantiate_quest_actions(quest, args.difficulty, wild_list, tr_list)
            print_actions(quest.start, wild_list)

            skel = extract_quest_summary(quest, args.difficulty, wild_list, tr_list)

            # Print QDS information
            print(f"\nQuest Difficulty Score (QDS): {skel['qds']}/100")
            print("Breakdown:")
            print(f"- Base Difficulty: {args.difficulty}/3")
            print(f"- Pokémon Encounters: {skel['qds_components']['pokemon_count']}")
            print(f"- Steps: {skel['qds_components']['steps']}")
            print(f"- Battle Strength: {skel['qds_components']['battle_strength']} (total levels)")
            print(f"- Special Conditions: {skel['qds_components']['special_conditions']}")
            print(f"- Avg Catch Difficulty: {skel['qds_components']['avg_catch_difficulty']}%")
            print(f"- Avg Base Stats: {skel['qds_components']['avg_base_stats']}%")

            main_nodes = extract_main_nodes(quest)
            for idx, node in enumerate(main_nodes):
                actions = node.actions
                step_prompt = f"""
    You are writing step‐by‐step Pokémon‐style NPC dialogue for HeartGold/SoulSilver.

    Trainer Context:
    - Location: Route 32
    - Time: Day
    - Wild Pokémon in the area: {wild_summary}
    - Trainers nearby:
    {trainer_info}

    Quest Step #{idx+1}:
    Actions the player must complete: {', '.join(actions)}

    Guidelines:
    - Keep each step focused:
    1. Initial request (“Please do X.”)
    2. Optional encouragement (“You’re doing great!”)
    3. Completion & reward (“Thanks—here’s your Y.”)
    - Use concise, in‐character NPC phrasing.
    - Do not mention difficulty or any out‐of‐band details.

    Respond only with the NPC’s lines.
    """
                resp = model.generate_content(step_prompt)
                print(f"\n-- Main Quest Step {idx+1} --\n", resp.text.strip())

                rom_df = pd.read_csv('rom_scripts.csv')

                # 2. Define a sample quest skeleton (in practice, import from your generator)
                quest_skeleton = {
                    "starter": "talkToHumanNPC",
                    "steps": [
                        "battle wild Hoothoot (Lv 6–8)",
                        "collect Potion",
                        "battle Trainer Youngster Albert with Rattata (Lv 6), Zubat (Lv 8)"
                    ],
                    "completion": "receiveItem"
                }

                # 3. Define a mapping from action keywords to ROM command sequences
                #    (you can refine this to use rom_df lookups instead of hardcoding)
                command_map = {
                    "talkToHumanNPC": [
                        "LockAll",
                        "FacePlayer",
                        "Message 20",       # Text line 20 = Help me research Hoothoot?
                        "WaitButton",
                        "CloseMessage",
                        "ReleaseAll"
                    ],
                    "battle wild": [
                        "LockAll",
                        "FacePlayer",
                        "Message 22",       # e.g. “Found a Hoothoot yet?”
                        "WaitButton",
                        "CloseMessage",
                        "ReleaseAll"
                    ],
                    "collect": [
                        "LockAll",
                        "FacePlayer",
                        "Message 24",       # e.g. “Thanks for all your help!”
                        "WaitButton",
                        "CloseMessage",
                        "ReleaseAll"
                    ],
                    "battle Trainer": [
                        "LockAll",
                        "FacePlayer",
                        "Message 26",       # e.g. “My research is complete!”
                        "WaitButton",
                        "CloseMessage",
                        "ReleaseAll"
                    ],
                    "receiveItem": [
                        "LockAll",
                        "FacePlayer",
                        "Message 27",       # e.g. “Here is your reward!”
                        "WaitButton",
                        "CloseMessage",
                        "ReleaseAll"
                    ],
                }

                # 4. Build the full script text
                lines = []
                script_id = 10
                lines.append(f"Script {script_id}:")
                for phase, action in [("Starter", quest_skeleton["starter"])] + \
                                    [(f"Step {i+1}", a) for i, a in enumerate(quest_skeleton["steps"])] + \
                                    [("Completion", quest_skeleton["completion"])]:
                    # Determine which command sequence to use based on action keyword
                    key = next((k for k in command_map if action.startswith(k)), None)
                    if key:
                        for cmd in command_map[key]:
                            lines.append(f"    {cmd}")
                    else:
                        lines.append(f"    # No mapping for '{action}'")
                lines.append("End\n\n")

                # 5. Write to text file
                f.write("\n".join(lines))

                print(f"ROM script written to {output_path}")