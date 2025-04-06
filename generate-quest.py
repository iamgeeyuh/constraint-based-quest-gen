import random

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
        {"id": "receivePokemon", "weight": 0.8},
        {"id": "receiveItem", "weight": 1.0},
        {"id": "returnToNPC", "weight": 1.0},
        {"id": "unlockArea", "weight": 0.6}
    ]
}

adjacency = {
    "talkToHumanNPC": ["collectXItems", "battleXPokemon", "findXPokemon", "deliverXItem", "escortXNPC"],
    "talkToPokemonNPC": ["followXPokemon", "encounterXPokemon", "findXPokemon"],
    "findClueItem": ["findXPokemon", "encounterXPokemon", "useHMAbility"],
    "overhearConversation": ["battleXTrainers", "encounterXPokemon"],
    "readSignOrNote": ["collectXItems", "findXPokemon"],
    "collectXItems": ["deliverXItem", "returnToNPC", "receiveItem"],
    "battleXPokemon": ["receivePokemon", "returnToNPC"],
    "battleXTrainers": ["receiveItem", "returnToNPC"],
    "encounterXPokemon": ["battleXPokemon", "followXPokemon", "receivePokemon"],
    "findXPokemon": ["battleXPokemon", "receivePokemon"],
    "followXPokemon": ["findXPokemon", "returnToNPC"],
    "deliverXItem": ["returnToNPC", "receiveItem"],
    "escortXNPC": ["returnToNPC", "receiveItem"],
    "useHMAbility": ["findXPokemon", "collectXItems"],
    "returnToNPC": ["receiveItem", "receivePokemon", "unlockArea"],
    "receiveItem": [],
    "receivePokemon": [],
    "unlockArea": []
}

def weighted_choice(tiles):
    ids = [t["id"] for t in tiles]
    weights = [t["weight"] for t in tiles]
    return random.choices(ids, weights=weights, k=1)[0]

def get_tile_data(tile_id):
    for group in quest_tiles.values():
        for tile in group:
            if tile["id"] == tile_id:
                return tile
    return None

def generate_quest(max_length=5):
    quest = []

    current = weighted_choice(quest_tiles["starters"])
    quest.append(current)

    while len(quest) < max_length:
        options = adjacency.get(current, [])
        if not options:
            break

        possible_tiles = [get_tile_data(t) for t in options if get_tile_data(t) is not None]
        if not possible_tiles:
            break

        next_tile = weighted_choice(possible_tiles)
        quest.append(next_tile)
        current = next_tile

        if current in [tile["id"] for tile in quest_tiles["completions"]]:
            break

    return quest


if __name__ == "__main__":
    for i in range(3):
        print(f"Generated Quest #{i + 1}:")
        path = generate_quest()
        for step in path:
            print(f"  → {step}")
        print()