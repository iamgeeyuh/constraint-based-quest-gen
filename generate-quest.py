
import random
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# Gemini Configuration
# -----------------------------

genai.configure(api_key=os.getenv('API_KEY'))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# -----------------------------
# Global Quest Structure Constraints
# -----------------------------

MAX_MAIN_STEPS = 4
MIN_MAIN_STEPS = 2
MAX_TASKS_PER_NODE = 2
SIDE_QUEST_CHANCE = 0.7
DECISION_FORK_CHANCE = 0.3
DIFFICULTY = 1

if DIFFICULTY == 1:  
    MIN_MAIN_STEPS = 2
    MAX_MAIN_STEPS = 3
    SIDE_QUEST_CHANCE = 0.4
    DECISION_FORK_CHANCE = 0.1
elif DIFFICULTY == 2:  
    MIN_MAIN_STEPS = 3
    MAX_MAIN_STEPS = 5
    SIDE_QUEST_CHANCE = 0.6
    DECISION_FORK_CHANCE = 0.3
else: 
    MIN_MAIN_STEPS = 4
    MAX_MAIN_STEPS = 6
    SIDE_QUEST_CHANCE = 0.8
    DECISION_FORK_CHANCE = 0.5

# -----------------------------
# Quest Tile Definitions
# -----------------------------

quest_tiles = {
    "starters": [
        {"id": "talkToHumanNPC", "weight": 1.0},
        {"id": "talkToPokemonNPC", "weight": 0.8},
        {"id": "findClueItem", "weight": 0.7},
        {"id": "overhearConversation", "weight": 0.5},
        {"id": "readSignOrNote", "weight": 0.6}
    ],
    "steps": [
        {"id": "collectXItems", "weight": 1.0},
        {"id": "battleXPokemon", "weight": 1.0},
        {"id": "battleXTrainers", "weight": 0.9},
        {"id": "encounterXPokemon", "weight": 0.8},
        {"id": "findXPokemon", "weight": 1.0},
        {"id": "followXPokemon", "weight": 0.6},
        {"id": "deliverXItem", "weight": 0.9},
        {"id": "escortXNPC", "weight": 0.5},
        {"id": "useHMAbility", "weight": 0.7}
    ],
    "completions": [
        {"id": "receivePokemon", "weight": 0.5},
        {"id": "receiveMoney", "weight": 1.0},
        {"id": "receiveItem", "weight": 0.8},
        {"id": "returnToNPC", "weight": 1.0},
    ]
}

starter_adjacency = {
    "talkToHumanNPC": ["findClueItem", "readSignOrNote"],
    "talkToPokemonNPC": ["overhearConversation", "findClueItem"],
    "findClueItem": ["readSignOrNote"],
    "overhearConversation": ["readSignOrNote"],
    "readSignOrNote": []
}

step_adjacency = {
    "collectXItems": ["deliverXItem", "useHMAbility"],
    "battleXPokemon": ["encounterXPokemon", "battleXTrainers"],
    "battleXTrainers": ["battleXPokemon"],
    "encounterXPokemon": ["battleXPokemon", "followXPokemon"],
    "findXPokemon": ["followXPokemon"],
    "followXPokemon": ["escortXNPC"],
    "deliverXItem": ["useHMAbility"],
    "escortXNPC": ["useHMAbility"],
    "useHMAbility": []
}

completion_adjacency = {
    "receiveItem": [],
    "receivePokemon": [],
    "returnToNPC": ["receiveItem", "receivePokemon", "receiveMoney"],
    "receiveMoney": []
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

    def add_edge(self, from_id, to_id):
        self.nodes[from_id].children.append(self.nodes[to_id])

def generate_task_chain(start_task, max_len, adjacency):
    chain = [start_task]
    current = start_task
    while len(chain) < max_len:
        next_options = adjacency.get(current, [])
        if not next_options:
            break
        current = random.choice(next_options)
        chain.append(current)
    return chain

def assign_tile_actions(node):
    if node.type == "completion" or node.type == "decision":
        pool = quest_tiles["completions"]
        adjacency = completion_adjacency
        max_len = 1
    elif node.type == "starter":
        pool = quest_tiles["starters"]
        adjacency = starter_adjacency
        max_len = MAX_TASKS_PER_NODE
    elif node.type == "step":
        pool = quest_tiles["steps"]
        adjacency = step_adjacency
        max_len = MAX_TASKS_PER_NODE
    else:
        return

    start_task = random.choices(
        [t["id"] for t in pool],
        weights=[t["weight"] for t in pool]
    )[0]

    node.actions = generate_task_chain(start_task, max_len, adjacency)

def populate_quest(graph):
    for node in graph.nodes.values():
        assign_tile_actions(node)

def generate_dynamic_quest_structure():
    q = QuestGraph()
    start = q.add_node("start", "starter")
    prev = start
    main_steps = []

    num_main_steps = random.randint(MIN_MAIN_STEPS, MAX_MAIN_STEPS)
    for i in range(num_main_steps):
        step = q.add_node(f"main{i+1}", "step")
        main_steps.append(step)
        q.add_edge(prev.id, step.id)
        prev = step

    if random.random() < SIDE_QUEST_CHANCE:
        base = random.choice(main_steps)
        side1 = q.add_node("side1", "step", optional=True)
        q.add_edge(base.id, side1.id)
        if random.random() < 0.5:
            side2 = q.add_node("side2", "step", optional=True)
            q.add_edge(side1.id, side2.id)
            prev_side = side2
        else:
            prev_side = side1
        side_complete = q.add_node("sideComplete", "completion", optional=True)
        q.add_edge(prev_side.id, side_complete.id)

    if random.random() < DECISION_FORK_CHANCE:
        decision = q.add_node("mainDecision", "step")
        q.add_edge(prev.id, decision.id)
        choiceA = q.add_node("decisionA", "completion")
        choiceB = q.add_node("decisionB", "completion")
        q.add_edge(decision.id, choiceA.id)
        q.add_edge(decision.id, choiceB.id)
    else:
        final_complete = q.add_node("complete", "completion")
        q.add_edge(prev.id, final_complete.id)

    return q

def print_actions(node, visited=None, prefix="", is_last=True):
    if visited is None:
        visited = set()
    if node.id in visited:
        return
    visited.add(node.id)
    connector = "└── " if is_last else "├── "
    opt = " (optional)" if node.optional else ""
    print(prefix + connector + f"{node.id}{opt}")
    action_prefix = prefix + ("    " if is_last else "│   ")
    for action in node.actions:
        print(action_prefix + f"• {action}")
    for i, child in enumerate(node.children):
        print_actions(child, visited, action_prefix, i == len(node.children) - 1)

def extract_quest_summary(graph):
    starter_action = graph.start.actions[0] if graph.start.actions else "talkToHumanNPC"
    step_nodes = [n for n in graph.nodes.values() if n.type == "step"]
    steps = [n.actions[0] if n.actions else "battleXPokemon" for n in step_nodes]
    completion_nodes = [n for n in graph.nodes.values() if n.type == "completion"]
    completion = completion_nodes[0].actions[0] if completion_nodes and completion_nodes[0].actions else "receiveItem"
    return {"starter": starter_action, "steps": steps, "completion": completion}

def extract_main_nodes(graph):
    """Returns a list of main quest nodes in order of appearance (excluding optional or side)."""
    main_nodes = []
    current = graph.start
    visited = set()
    while current and current.id not in visited:
        visited.add(current.id)
        if current.type == "step":
            main_nodes.append(current)
        children = [c for c in current.children if not c.optional]
        current = children[0] if children else None
    return main_nodes

# -----------------------------
# Load Route 32 Data
# -----------------------------

encounters_df = pd.read_csv("route_32_encounters.csv")
trainers_df = pd.read_csv("route_32_trainers.csv")

day_encounters = encounters_df[encounters_df["Time"] == "Day"]
wild_summary = ", ".join(
    f"{row['Pokemon']} (Lv {row['Level(s)']}, {row['Encounter Rate']})"
    for _, row in day_encounters.iterrows()
)

trainer_lines = []
grouped = trainers_df.groupby("Trainer Name").agg({
    "Pokemon": lambda x: ", ".join(x),
    "Level(s)": lambda x: ", ".join(str(i) for i in x)
}).reset_index()

for _, row in grouped.head(3).iterrows():
    trainer_lines.append(f"{row['Trainer Name']} ({row['Pokemon']} at Lv {row['Level(s)']})")
trainer_info = "\n".join(trainer_lines)

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    for i in range(3):
        print(f"\n=== Generated Quest #{i + 1} ===")
        quest = generate_dynamic_quest_structure()
        populate_quest(quest)
        print_actions(quest.start)

        skeleton = extract_quest_summary(quest)

        prompt = f"""
You are writing quest dialogue for a side quest in Pokémon HeartGold/SoulSilver.

Trainer Context:
- Location: Route 32
- Time: Day
- Wild Pokémon in the area: {wild_summary}
- Trainers nearby:
{trainer_info}

Quest Skeleton:
- Starter: {skeleton['starter']}
- Step(s): {', '.join(skeleton['steps'])}
- Completion: {skeleton['completion']}

Quest Difficulty Level: {DIFFICULTY} (1 = easy, 2 = medium, 3 = hard)

Guidelines:
- If Difficulty 1 (Easy): Objectives should be simple and quick (ex: battle any wild Pokémon, deliver a common item, talk to an NPC).
- If Difficulty 2 (Medium): Objectives can require some exploration or minor battles (ex: catch a semi-common Pokémon, battle a few trainers).
- If Difficulty 3 (Hard): Objectives should be more challenging (ex: catch rare Pokémon, defeat stronger trainers, find hidden items).

Write character dialogue in Pokémon NPC style:
1. Initial greeting and request
2. In-quest encouragement (optional)
3. Completion and reward

Output only the NPC’s lines.
"""

        response = model.generate_content(prompt)
        
        print("\n--- NPC Dialogue ---\n")
        main_nodes = extract_main_nodes(quest)
        for idx, node in enumerate(main_nodes):
            actions = node.actions
            prompt = f"""You are writing step-by-step Pokémon-style NPC dialogue from HeartGold/SoulSilver.

The player returns to the same NPC after each step is completed to receive the next task. Dialogue should flow as a single narrative from the same character.

Trainer Context:
- Location: Route 32
- Time: Day
- Wild Pokémon in the area: {wild_summary}
- Trainers nearby:
{trainer_info}

Quest Step #{idx + 1}:
Actions the player must complete: {', '.join(actions)}

Quest Difficulty Level: {DIFFICULTY} (1 = easy, 2 = medium, 3 = hard)

Guidelines:
- If Difficulty 1 (Easy): Objectives should be simple and quick (ex: battle any wild Pokémon, deliver a common item, talk to an NPC).
- If Difficulty 2 (Medium): Objectives can require some exploration or minor battles (ex: catch a semi-common Pokémon, battle a few trainers).
- If Difficulty 3 (Hard): Objectives should be more challenging (ex: catch rare Pokémon, defeat stronger trainers, find hidden items).

Write only the dialogue for this one step (as if the player has just returned from the previous one).
1. Initial request
2. In-quest encouragement (optional)
3. Completion and reward

Respond only with the NPC’s lines.
"""
            response = model.generate_content(prompt)
            print(f"\n-- Main Quest Step {idx + 1} --\n")
            print(response.text.strip())
    
