"""Universal prompt fragments shared across all Jericho games."""

ROLE_AND_GOAL = (
    "You are playing a classic interactive fiction text adventure game. "
    "You interact with the game world by typing short text commands. "
    "The game parser reads your command, updates the world, and describes "
    "what happens.\n\n"
    "Your goal is to maximize your score. Read the game's text descriptions "
    "carefully \u2014 they contain clues about what to do, where to go, and "
    "what items are important."
)

ACTION_STRUCTURE = (
    "## Command Structure\n\n"
    "Commands follow three patterns:\n\n"
    "1. VERB \u2014 a single action with no object.\n"
    "   Examples: look, inventory, wait, north, south, east, west, up, down\n\n"
    "2. VERB [PREPOSITION] OBJECT \u2014 an action targeting one object. "
    "The preposition is optional depending on the verb.\n"
    "   Examples: take lamp, open mailbox, examine sword, "
    "look at painting, climb up tree\n\n"
    "3. VERB OBJECT PREPOSITION OBJECT \u2014 an action involving two objects.\n"
    "   Examples: put egg in case, attack troll with sword, "
    "unlock door with key, take lamp from table\n\n"
    "Only use command words, directions, and prepositions listed in the "
    "reference sections below. Keep commands short (2-4 words)."
)

PREPOSITIONS = (
    "## Prepositions\n\n"
    "Use these words to connect verbs and objects:\n"
    "in, on, at, to, from, with, under, over, through, into, onto, off, "
    "up, down, out, about, for, behind, across, around"
)

GAME_MECHANICS = (
    "## Game Mechanics\n\n"
    "Inventory: You can carry items. Type \"inventory\" (or \"i\") to see "
    "what you have. There is a limit on how many items you can carry \u2014 "
    "drop items you do not need.\n\n"
    "Containers: Bags, boxes, chests, and cases hold items. They must be "
    "open before you can put things in or take things out. Use \"open X\", "
    "\"put Y in X\", \"take Y from X\".\n\n"
    "Darkness: Some areas are dark. You need a light source (lamp, torch, "
    "candles) to see. Without light, you cannot examine objects and may be "
    "in danger. Light sources can run out.\n\n"
    "State persistence: The game world remembers your actions. Doors you "
    "open stay open. Items stay where you drop them. Characters react to "
    "what you have done.\n\n"
    "Parser tips: If the game says it does not understand a command, try a "
    "different verb or construction from the reference above. If it says "
    "you cannot do something, the command was understood but the action is "
    "not possible right now \u2014 you may need a different item, to be in "
    "a different location, or to solve a prerequisite puzzle first."
)
