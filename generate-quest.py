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

QDS_WEIGHTS = {
    'difficulty': 0.4,      # v - Base difficulty setting
    'pokemon': 0.2,         # w - Number of Pokémon involved
    'steps': 0.15,          # x - Number of quest steps
    'battles': 0.15,        # y - Battle strength (sum of levels)
    'conditions': 0.1       # z - Special conditions (optional steps, etc)
}

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
        wild.append({
            "name":         r["Pokemon"],
            "level_range":  r["Level(s)"],
            "enc_rate":     float(r["Encounter Rate"].strip("%")) / 100
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
    if diff == 1:
        pool = [w for w in wild_list if w["enc_rate"] >= 0.3]
    elif diff == 2:
        pool = [w for w in wild_list if 0.1 <= w["enc_rate"] < 0.3]
    else:
        pool = [w for w in wild_list if w["enc_rate"] < 0.1]
    return random.choice(pool or wild_list)

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

def calculate_qds(graph, difficulty, wild_list, tr_list):
    """Calculate Quest Difficulty Score (QDS)"""
    # Extract basic metrics
    main_nodes = extract_main_nodes(graph)
    num_steps = len(main_nodes)
    
    # Count Pokémon encounters
    pokemon_count = sum(
        1 for node in graph.nodes.values() 
        if any("Pokemon" in a or "Pokémon" in a for a in node.actions)
    )
    
    # Calculate battle strength
    battle_strength = 0
    for node in graph.nodes.values():
        for action in node.actions:
            if "battle" in action.lower():
                if "wild" in action:
                    # Extract wild Pokémon level (take max if range)
                    level_str = action.split("Lv ")[1].split(")")[0]
                    if "–" in level_str:
                        level = int(level_str.split("–")[1])
                    else:
                        level = int(level_str)
                    battle_strength += level
                elif "Trainer" in action:
                    # Sum all trainer Pokémon levels
                    levels = [
                        int(lv.split(")")[0]) 
                        for lv in action.split("Lv ")[1:]
                    ]
                    battle_strength += sum(levels)
    
    # Count special conditions (optional nodes, complex tasks)
    special_conds = sum(
        1 for node in graph.nodes.values() 
        if node.optional or len(node.actions) > 1
    )
    
    # Calculate weighted score
    score = (
        QDS_WEIGHTS['difficulty'] * difficulty +
        QDS_WEIGHTS['pokemon'] * pokemon_count +
        QDS_WEIGHTS['steps'] * num_steps +
        QDS_WEIGHTS['battles'] * (battle_strength / 50) +  # Normalized by dividing by 50
        QDS_WEIGHTS['conditions'] * special_conds
    )
    
    # Scale to 0-100 range
    qds = min(100, max(0, int(score * 20)))
    
    return {
        'qds': qds,
        'components': {
            'difficulty': difficulty,
            'pokemon_count': pokemon_count,
            'steps': num_steps,
            'battle_strength': battle_strength,
            'special_conditions': special_conds
        }
    }

def extract_quest_summary(graph, difficulty, wild_list, tr_list):
    starter = graph.start.actions[0] if graph.start.actions else ""
    steps = [n.actions[0] for n in graph.nodes.values() if n.type == "step"]
    comp = [n.actions[0] for n in graph.nodes.values() if n.type == "completion"]
    qds_data = calculate_qds(graph, difficulty, wild_list, tr_list)
    
    return {
        "starter": starter,
        "steps": steps,
        "completion": comp[0] if comp else "",
        "qds": qds_data['qds'],
        "qds_components": qds_data['components']
    }

# -----------------------------
# Printing & Extraction
# -----------------------------
def print_actions(node, visited=None, prefix="", is_last=True):
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
        print(child_prefix + f"• {action}")
    for i, ch in enumerate(node.children):
        print_actions(ch, visited, child_prefix, i == len(node.children) - 1)

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
        default="route_32_encounters.csv"
    )
    parser.add_argument(
        "--trainers",
        default="route_32_trainers.csv"
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
            print_actions(quest.start)

            skel = extract_quest_summary(quest, args.difficulty, wild_list, tr_list)

            # Print QDS information
            print(f"\nQuest Difficulty Score (QDS): {skel['qds']}/100")
            print("Breakdown:")
            print(f"- Base Difficulty: {args.difficulty}/3")
            print(f"- Pokémon Encounters: {skel['qds_components']['pokemon_count']}")
            print(f"- Steps: {skel['qds_components']['steps']}")
            print(f"- Battle Strength: {skel['qds_components']['battle_strength']} (total levels)")
            print(f"- Special Conditions: {skel['qds_components']['special_conditions']}")

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