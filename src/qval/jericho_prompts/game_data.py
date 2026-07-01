"""Per-game metadata for Jericho text adventure environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JerichoGameData:
    """Static metadata for a single Jericho game."""

    display_name: str
    genre: str
    premise: str
    max_score: int
    scoring_description: str


GAME_DATA: dict[str, JerichoGameData] = {
    "905": JerichoGameData(
        display_name="9:05",
        genre="slice-of-life",
        premise=(
            "You have overslept and must get ready and drive to work on "
            "time. The entire game is a tightly scripted morning routine."
        ),
        max_score=1,
        scoring_description=(
            "The single point is awarded for successfully completing your "
            "morning routine and getting to work."
        ),
    ),
    "acorncourt": JerichoGameData(
        display_name="Acorn Court",
        genre="puzzle",
        premise=(
            "You find yourself in a court yard and must figure out a series "
            "of mechanical puzzles involving a tennis ball machine, a bucket, "
            "and acorns to find a key and escape."
        ),
        max_score=30,
        scoring_description=(
            "Points for collecting key items and solving the mechanical "
            "puzzles in the court yard."
        ),
    ),
    "advent": JerichoGameData(
        display_name="Adventure",
        genre="cave crawl / treasure hunt",
        premise=(
            "Explore Colossal Cave, a vast underground cavern system. Find "
            "the 15 treasures hidden throughout the cave and bring them back "
            "to the well house to score points. Beware of dwarves, a snake, "
            "and a pirate."
        ),
        max_score=350,
        scoring_description=(
            "Points for finding the 15 treasures in the cave and depositing "
            "them in the well house building. Picking up a treasure earns "
            "points; depositing it earns more."
        ),
    ),
    "adventureland": JerichoGameData(
        display_name="Adventureland",
        genre="treasure hunt",
        premise=(
            "Explore a fantasy world to find 13 hidden treasures (marked "
            "with *) and return them to score points. The world includes "
            "forests, a swamp, underground caverns, and a lake."
        ),
        max_score=100,
        scoring_description=(
            "Points for finding the 13 treasures (marked with *) and "
            "storing them."
        ),
    ),
    "afflicted": JerichoGameData(
        display_name="Afflicted",
        genre="horror / exploration",
        premise=(
            "You are a city sanitarian inspecting a seedy restaurant in a "
            "bad neighborhood. Explore the building, collect evidence items, "
            "and uncover what is really going on."
        ),
        max_score=75,
        scoring_description=(
            "Points for finding evidence items and reaching new areas of "
            "the building during your inspection."
        ),
    ),
    "anchor": JerichoGameData(
        display_name="Anchorhead",
        genre="horror / mystery",
        premise=(
            "You have just moved to the coastal town of Anchorhead, "
            "Massachusetts. Investigate the town's dark history and uncover "
            "a horrific conspiracy inspired by Lovecraftian horror across "
            "multiple days."
        ),
        max_score=100,
        scoring_description=(
            "Points from exploring Anchorhead, interacting with characters, "
            "finding items, and solving puzzles across the story."
        ),
    ),
    "awaken": JerichoGameData(
        display_name="Awaken",
        genre="puzzle / exploration",
        premise=(
            "You wake up in mud with no memory. Explore your surroundings "
            "to discover who you are and what happened. Points are earned "
            "by reaching key locations and making discoveries."
        ),
        max_score=50,
        scoring_description=(
            "Points for reaching key locations and making important "
            "discoveries about your identity."
        ),
    ),
    "balances": JerichoGameData(
        display_name="Balances",
        genre="fantasy / puzzle",
        premise=(
            "You are in a ramshackle hut with a spell book and a magic "
            "burin. Explore a small fantasy world, collect items, and solve "
            "puzzles involving magic spells."
        ),
        max_score=51,
        scoring_description=(
            "Points from collecting items, exploring locations, casting "
            "spells, and solving puzzles."
        ),
    ),
    "ballyhoo": JerichoGameData(
        display_name="Ballyhoo",
        genre="mystery",
        premise=(
            "After watching a circus show, you become entangled in a "
            "mystery involving a kidnapping. Explore behind the scenes of "
            "the circus, gather clues, and solve the case."
        ),
        max_score=200,
        scoring_description=(
            "Points for reaching important circus locations and progressing "
            "through the kidnapping mystery."
        ),
    ),
    "curses": JerichoGameData(
        display_name="Curses",
        genre="fantasy / exploration",
        premise=(
            "Explore a sprawling old house and its many connected "
            "locations. The game involves collecting items and solving "
            "puzzles across a very large map with many rooms."
        ),
        max_score=550,
        scoring_description=(
            "Points primarily from exploring the many rooms of the house "
            "and solving puzzles. The game has a very large map."
        ),
    ),
    "cutthroat": JerichoGameData(
        display_name="Cutthroat",
        genre="adventure / mystery",
        premise=(
            "Your partner has told you about a sunken treasure. Explore "
            "the island, dive underwater, and recover the treasure while "
            "dealing with treachery and danger."
        ),
        max_score=250,
        scoring_description=(
            "Points for reaching key story locations and progressing "
            "toward the sunken treasure."
        ),
    ),
    "deephome": JerichoGameData(
        display_name="Deephome",
        genre="fantasy / dungeon crawl",
        premise=(
            "You are a dwarf on a mission from the King, carrying a "
            "lantern through underground passages. Explore caverns, solve "
            "puzzles, and complete your quest."
        ),
        max_score=300,
        scoring_description=(
            "Points from exploring underground passages, solving puzzles, "
            "and manipulating mechanisms."
        ),
    ),
    "detective": JerichoGameData(
        display_name="Detective",
        genre="mystery",
        premise=(
            "The Chief gives you a case: a murder at the Mayor's house. "
            "Investigate the crime scene, gather evidence across town, and "
            "track down the killer. Points are earned for visiting "
            "locations and finding clues."
        ),
        max_score=360,
        scoring_description=(
            "Points are earned for almost every new location visited and "
            "evidence item found. Very high scoring density."
        ),
    ),
    "dragon": JerichoGameData(
        display_name="Dragon",
        genre="fantasy",
        premise=(
            "The council sends you on a quest to deal with a dragon "
            "threatening the land. Explore the world, gather allies and "
            "items, and confront the dragon."
        ),
        max_score=25,
        scoring_description=(
            "Points from interacting with characters, giving items, and "
            "progressing the quest."
        ),
    ),
    "enchanter": JerichoGameData(
        display_name="Enchanter",
        genre="fantasy / magic",
        premise=(
            "You are a novice Enchanter sent to defeat the evil warlock "
            "Krill. Explore the land, learn new spells from scrolls, and "
            "use magic to overcome obstacles and defeat Krill."
        ),
        max_score=400,
        scoring_description=(
            "Points from reaching new areas, solving puzzles, learning "
            "spells, and ultimately defeating Krill."
        ),
    ),
    "enter": JerichoGameData(
        display_name="Enter the Dark",
        genre="school / horror",
        premise=(
            "You are a student at a middle school. Explore the school and "
            "its surroundings, interact with people, and earn points by "
            "giving items to the right characters."
        ),
        max_score=20,
        scoring_description=(
            "Points primarily from giving items to the right characters."
        ),
    ),
    "gold": JerichoGameData(
        display_name="Gold",
        genre="fairy tale / puzzle",
        premise=(
            "You find yourself in an enchanted forest wearing a dress and "
            "a wristwatch. Navigate a fairy-tale world solving puzzles, "
            "interacting with characters, and exploring locations."
        ),
        max_score=100,
        scoring_description=(
            "Points from exploring locations, solving puzzles, depositing "
            "items, and interacting with fairy-tale characters."
        ),
    ),
    "hhgg": JerichoGameData(
        display_name="Hitchhiker's Guide to the Galaxy",
        genre="science fiction / comedy",
        premise=(
            "Based on Douglas Adams' novel. Earth is about to be "
            "demolished. Survive the destruction, hitchhike through space, "
            "collect key items, and navigate absurd puzzles aboard the "
            "Heart of Gold."
        ),
        max_score=400,
        scoring_description=(
            "Points for reaching key story locations and collecting "
            "important items like the Babel fish and the towel."
        ),
    ),
    "hollywood": JerichoGameData(
        display_name="Hollywood Hijinx",
        genre="treasure hunt / puzzle",
        premise=(
            "Your late Aunt Hildegarde has left you her mansion, but only "
            "if you can find all the treasures hidden inside. Explore the "
            "house and grounds to find Uncle Buddy's hidden prizes."
        ),
        max_score=150,
        scoring_description=(
            "Points for finding Uncle Buddy's hidden treasures throughout "
            "the mansion and grounds."
        ),
    ),
    "huntdark": JerichoGameData(
        display_name="Hunt the Dark",
        genre="exploration / puzzle",
        premise=(
            "You are exploring a branching cave system with a crossbow "
            "and a lamp lens. Navigate the dark caves to reach your goal."
        ),
        max_score=1,
        scoring_description=(
            "The single point is for reaching the end of the cave system."
        ),
    ),
    "infidel": JerichoGameData(
        display_name="Infidel",
        genre="adventure / exploration",
        premise=(
            "You are an archaeologist searching for a hidden Egyptian "
            "pyramid in the desert. Find the pyramid, navigate its traps "
            "and puzzles, and recover the ancient treasures inside."
        ),
        max_score=400,
        scoring_description=(
            "Points from exploring the pyramid, collecting treasures, "
            "solving traps, and reaching deeper chambers."
        ),
    ),
    "inhumane": JerichoGameData(
        display_name="Inhumane",
        genre="puzzle / escape",
        premise=(
            "You wake up in a tent with a hangover during an expedition. "
            "Explore the camp and surrounding area, solve puzzles, and "
            "figure out what happened."
        ),
        max_score=90,
        scoring_description=(
            "Points from exploring the camp, solving puzzles, and "
            "reaching key milestones."
        ),
    ),
    "jewel": JerichoGameData(
        display_name="Jewel of Knowledge",
        genre="dungeon crawl",
        premise=(
            "You carry a silver sword through the deep layers of the "
            "Earth's crust. Navigate underground passages, solve puzzles, "
            "and search for the Jewel of Knowledge."
        ),
        max_score=90,
        scoring_description=(
            "Points from exploring underground layers, solving puzzles, "
            "and interacting with the environment."
        ),
    ),
    "karn": JerichoGameData(
        display_name="Karn the Barbarian",
        genre="science fiction / fantasy",
        premise=(
            "You are a time-traveling adventurer who must explore a "
            "planet, solve puzzles, and complete your mission. Your TARDIS "
            "has brought you somewhere new."
        ),
        max_score=170,
        scoring_description=(
            "Points from collecting items, depositing them, solving "
            "puzzles, and interacting with characters."
        ),
    ),
    "library": JerichoGameData(
        display_name="All Quiet on the Library Front",
        genre="puzzle / comedy",
        premise=(
            "You are in a library with your ID card. Explore the library, "
            "interact with the staff and patrons, and solve a series of "
            "small puzzles to earn points."
        ),
        max_score=30,
        scoring_description=(
            "Points from interacting with library staff, finding items, "
            "and solving the library's puzzles."
        ),
    ),
    "loose": JerichoGameData(
        display_name="Loose",
        genre="surreal / puzzle",
        premise=(
            "Something is out there. Explore a surreal countryside, "
            "interact with strange objects and creatures, and solve "
            "puzzles in this unusual adventure."
        ),
        max_score=50,
        scoring_description=(
            "Points from exploring the surreal landscape, collecting "
            "items, and solving puzzles."
        ),
    ),
    "lostpig": JerichoGameData(
        display_name="Lost Pig",
        genre="fantasy / comedy",
        premise=(
            "You are Grunk, an orc. Your pig has escaped and your boss "
            "blames you. Find the lost pig in an underground complex, "
            "interact with a gnome, and solve puzzles. Written entirely "
            "in Grunk's broken English."
        ),
        max_score=7,
        scoring_description=(
            "Each point marks a major puzzle milestone: listening, "
            "examining, giving items, unlocking, and catching the pig."
        ),
    ),
    "ludicorp": JerichoGameData(
        display_name="The Ludicorp Mystery",
        genre="comedy / exploration",
        premise=(
            "It is nearly Christmas and Ludicorp still has not released "
            "their game. Investigate the Ludicorp offices to find out "
            "what is going on. Points come from exploring rooms and "
            "discovering clues."
        ),
        max_score=150,
        scoring_description=(
            "Points for exploring the Ludicorp offices. Almost every new "
            "room visited earns points."
        ),
    ),
    "lurking": JerichoGameData(
        display_name="The Lurking Horror",
        genre="horror",
        premise=(
            "You are a student at a university on a dark and stormy "
            "night. Something sinister lurks in the campus buildings. "
            "Explore, solve puzzles, and survive the horror."
        ),
        max_score=100,
        scoring_description=(
            "Points from exploring the campus, collecting items, and "
            "solving puzzles while surviving the horror."
        ),
    ),
    "moonlit": JerichoGameData(
        display_name="Moonlit Tower",
        genre="puzzle",
        premise=(
            "You find yourself in a stone chamber wearing armor and "
            "silks. Solve the puzzle of this single, atmospheric "
            "location to escape."
        ),
        max_score=1,
        scoring_description=(
            "The single point is for solving the puzzle and escaping "
            "the stone chamber."
        ),
    ),
    "murdac": JerichoGameData(
        display_name="Murdac",
        genre="cave crawl / treasure hunt",
        premise=(
            "A text adventure in the tradition of Colossal Cave. Explore "
            "underground caves, collect treasures, solve puzzles, and "
            "deposit items for points."
        ),
        max_score=250,
        scoring_description=(
            "Points from collecting treasures and depositing them, plus "
            "reaching new cave locations. Score can decrease for bad "
            "actions."
        ),
    ),
    "night": JerichoGameData(
        display_name="Night at the Computer Center",
        genre="puzzle / comedy",
        premise=(
            "You are a site operator at a university computer center "
            "during a probationary period. Something has gone wrong with "
            "the computers. Fix the problems before morning."
        ),
        max_score=10,
        scoring_description=(
            "Points for fixing key computer problems and reaching "
            "milestones at the computer center."
        ),
    ),
    "omniquest": JerichoGameData(
        display_name="OmniQuest",
        genre="adventure / puzzle",
        premise=(
            "You are bored and lying in bed. Get up, explore the world, "
            "and solve puzzles. A lighthearted adventure that rewards "
            "exploration and interaction."
        ),
        max_score=50,
        scoring_description=(
            "Points from exploring, examining objects, giving items, "
            "and solving puzzles."
        ),
    ),
    "partyfoul": JerichoGameData(
        display_name="Party Foul",
        genre="social / puzzle",
        premise=(
            "You are at a party. Navigate the social situation, interact "
            "with guests, and achieve your hidden objective."
        ),
        max_score=1,
        scoring_description=(
            "The single point is for achieving your social objective "
            "at the party."
        ),
    ),
    "pentari": JerichoGameData(
        display_name="Pentari",
        genre="fantasy / combat",
        premise=(
            "You awake in your quarters in the city of Bostwin with your "
            "Pentarian sword. Adventure through a fantasy world, fighting "
            "enemies and collecting items."
        ),
        max_score=70,
        scoring_description=(
            "Points from collecting items, defeating enemies, and "
            "progressing through the fantasy quest."
        ),
    ),
    "planetfall": JerichoGameData(
        display_name="Planetfall",
        genre="science fiction",
        premise=(
            "You are a lowly Ensign Seventh Class aboard a spaceship. "
            "After a disaster, you crash-land on a deserted planet. "
            "Explore the abandoned complex, solve puzzles, and survive."
        ),
        max_score=80,
        scoring_description=(
            "Points from reaching new areas of the abandoned complex "
            "and solving survival puzzles."
        ),
    ),
    "plundered": JerichoGameData(
        display_name="Plundered Hearts",
        genre="romance / adventure",
        premise=(
            "You are a young woman in the 17th century Caribbean. "
            "Navigate intrigue, romance, and danger on a pirate ship "
            "and island."
        ),
        max_score=25,
        scoring_description=(
            "Points from progressing through the story, reaching new "
            "locations, and navigating the intrigue."
        ),
    ),
    "reverb": JerichoGameData(
        display_name="Reverb",
        genre="slice of life / puzzle",
        premise=(
            "You are a pizza delivery worker having a crummy day. Explore "
            "the town, interact with people, and solve problems to turn "
            "your day around."
        ),
        max_score=50,
        scoring_description=(
            "Points from exploring the town, helping people, and solving "
            "problems during your delivery shift."
        ),
    ),
    "seastalker": JerichoGameData(
        display_name="Seastalker",
        genre="science fiction / adventure",
        premise=(
            "You are a young inventor who has built a submarine. A sea "
            "monster threatens the Aquadome. Use your submarine and "
            "inventions to investigate and deal with the threat."
        ),
        max_score=100,
        scoring_description=(
            "Points from exploring with the submarine, investigating "
            "the threat, and using your inventions."
        ),
    ),
    "sherlock": JerichoGameData(
        display_name="Sherlock",
        genre="mystery / detective",
        premise=(
            "You are Sherlock Holmes. The Crown Jewels have been stolen "
            "from the Tower of London. Investigate London, gather clues, "
            "question suspects, and solve the case before the deadline."
        ),
        max_score=100,
        scoring_description=(
            "Points from finding evidence, questioning suspects, "
            "visiting locations, and deducing the solution."
        ),
    ),
    "snacktime": JerichoGameData(
        display_name="Snack Time",
        genre="comedy / puzzle",
        premise=(
            "Your stomach is growling after hours in front of the TV. "
            "Get up, find food, and deal with your mischievous pet along "
            "the way."
        ),
        max_score=50,
        scoring_description=(
            "Points from key milestones in your quest for food and "
            "dealing with your pet."
        ),
    ),
    "sorcerer": JerichoGameData(
        display_name="Sorcerer",
        genre="fantasy / magic",
        premise=(
            "You are in a twisted forest at the start of a magical quest. "
            "Explore the land, find and cast spells, and complete your "
            "sorcerous mission. Part of the Enchanter trilogy."
        ),
        max_score=400,
        scoring_description=(
            "Points from reaching key locations, finding spells, and "
            "completing the magical quest."
        ),
    ),
    "spellbrkr": JerichoGameData(
        display_name="Spellbreaker",
        genre="fantasy / magic",
        premise=(
            "The most powerful Enchanter, you attend a Council meeting "
            "where magic seems to be failing. Investigate the collapse "
            "of magic, collect magical cubes, and save the world. Part "
            "of the Enchanter trilogy."
        ),
        max_score=600,
        scoring_description=(
            "Points primarily from collecting magical cubes. Each cube "
            "is worth many points."
        ),
    ),
    "spirit": JerichoGameData(
        display_name="Spiritwrak",
        genre="fantasy / exploration",
        premise=(
            "You are in a monastery chapel with a prayer book. Explore a "
            "large fantasy world, collect items, solve puzzles, and "
            "unravel the mystery of the Spirits."
        ),
        max_score=250,
        scoring_description=(
            "Points from exploring the world, collecting items, "
            "depositing them, and solving puzzles."
        ),
    ),
    "temple": JerichoGameData(
        display_name="Temple",
        genre="surreal / puzzle",
        premise=(
            "You are in a void, experiencing a recurring dream. Explore "
            "a mysterious temple environment, examine objects carefully, "
            "and solve puzzles to understand what is happening."
        ),
        max_score=35,
        scoring_description=(
            "Points from carefully examining objects, solving the "
            "temple's puzzles, and making discoveries."
        ),
    ),
    "theatre": JerichoGameData(
        display_name="Theatre",
        genre="horror / exploration",
        premise=(
            "You arrive at the lobby of a theatre for a night of horror. "
            "Explore the theatre building, solve puzzles, and survive "
            "whatever lurks inside."
        ),
        max_score=50,
        scoring_description=(
            "Points from exploring the theatre, collecting items, "
            "depositing them, and solving puzzles."
        ),
    ),
    "trinity": JerichoGameData(
        display_name="Trinity",
        genre="science fiction / fantasy",
        premise=(
            "You are a tourist in London when nuclear war breaks out. "
            "Escape through a magical door into a surreal garden "
            "connecting different nuclear test sites throughout history. "
            "Collect items and solve puzzles."
        ),
        max_score=100,
        scoring_description=(
            "Points from collecting items across the nuclear test site "
            "locations and solving puzzles."
        ),
    ),
    "tryst205": JerichoGameData(
        display_name="Tryst of Fate",
        genre="fantasy / puzzle",
        premise=(
            "You have just finished cleaning a house while wearing a "
            "gold watch. Explore a fantasy world of interconnected "
            "locations, solve puzzles, and uncover the story."
        ),
        max_score=350,
        scoring_description=(
            "Points from exploring locations, collecting items, solving "
            "puzzles, and interacting with the world."
        ),
    ),
    "weapon": JerichoGameData(
        display_name="Weapon",
        genre="science fiction",
        premise=(
            "You are in a control room with handcuffs. A single critical "
            "interaction determines your fate in this short, tense "
            "scenario."
        ),
        max_score=1,
        scoring_description=(
            "A single critical interaction determines the outcome."
        ),
    ),
    "wishbringer": JerichoGameData(
        display_name="Wishbringer",
        genre="fantasy / adventure",
        premise=(
            "The Princess is in danger and a trap has been sprung. "
            "Explore the town of Festeron, collect items, use the "
            "Wishbringer stone's magic, and rescue the Princess."
        ),
        max_score=100,
        scoring_description=(
            "Points from collecting items, using the Wishbringer stone, "
            "exploring Festeron, and rescuing the Princess."
        ),
    ),
    "yomomma": JerichoGameData(
        display_name="Yo Momma",
        genre="comedy / puzzle",
        premise=(
            "A comedy adventure involving yo momma jokes. Navigate a "
            "humorous world, interact with characters, and solve puzzles."
        ),
        max_score=35,
        scoring_description=(
            "Points from exploring, interacting with characters, and "
            "solving puzzles in the comedic world."
        ),
    ),
    "zenon": JerichoGameData(
        display_name="Escape from Starship Zenon",
        genre="science fiction / escape",
        premise=(
            "You have been imprisoned on a starship. Escape from your "
            "cell and navigate the ship to freedom."
        ),
        max_score=20,
        scoring_description=(
            "Points from escaping your cell and navigating through the "
            "starship to freedom."
        ),
    ),
    "zork1": JerichoGameData(
        display_name="Zork I",
        genre="dungeon crawl / treasure hunt",
        premise=(
            "Explore the Great Underground Empire beneath a white house. "
            "Find the 20 treasures scattered through underground caverns, "
            "mazes, and dangerous encounters, and deposit them in the "
            "trophy case in the living room."
        ),
        max_score=350,
        scoring_description=(
            "Points for finding the 20 treasures and depositing them in "
            "the trophy case in the living room of the white house. "
            "Picking up a treasure earns 1-15 points; placing it in the "
            "case earns 1-15 more."
        ),
    ),
    "zork2": JerichoGameData(
        display_name="Zork II",
        genre="dungeon crawl / treasure hunt",
        premise=(
            "Continue exploring the Great Underground Empire. Collect "
            "valuable items and treasures while dealing with the Wizard "
            "of Frobozz, who casts random spells at you, and other "
            "dangers."
        ),
        max_score=400,
        scoring_description=(
            "Points for collecting valuable items and treasures. Many "
            "items are worth 15-30 points when first picked up. "
            "Additional points for specific story milestones."
        ),
    ),
    "zork3": JerichoGameData(
        display_name="Zork III",
        genre="dungeon crawl / puzzle",
        premise=(
            "The final chapter of the Zork trilogy. Explore the deepest "
            "levels of the Underground Empire, solve puzzles involving "
            "time travel and moral choices, and confront the Dungeon "
            "Master."
        ),
        max_score=7,
        scoring_description=(
            "Each of the 7 points marks a major puzzle solution: jumping "
            "in the lake, touching the table, reaching the ledge, "
            "confronting the hooded figure, and navigating the endgame."
        ),
    ),
    "ztuu": JerichoGameData(
        display_name="Zork: The Undiscovered Underground",
        genre="dungeon crawl / treasure hunt",
        premise=(
            "Sent by the Grand Inquisitor into a new excavation, explore "
            "underground ruins with your sword and lantern. Find "
            "treasures and solve puzzles in this compact Zork adventure."
        ),
        max_score=100,
        scoring_description=(
            "Points for finding treasures in the underground ruins and "
            "storing them. Additional points for combat victories and "
            "reaching key areas."
        ),
    ),
}


def get_game_data(game_name: str) -> JerichoGameData:
    """Return per-game data, falling back to auto-generated if unknown."""
    if game_name in GAME_DATA:
        return GAME_DATA[game_name]
    return _auto_generate(game_name)


def _auto_generate(game_name: str) -> JerichoGameData:
    """Generate minimal game data for an unknown game."""
    max_score = 0
    try:
        from jericho import game_info  # type: ignore[import-untyped]

        info = getattr(game_info, game_name, None)
        if info is None:
            for attr in dir(game_info):
                candidate = getattr(game_info, attr)
                if isinstance(candidate, dict) and candidate.get("name") == game_name:
                    info = candidate
                    break
        if isinstance(info, dict):
            from jericho import FrotzEnv  # type: ignore[import-untyped]

            # Try to get max_score from the bindings if ROM is loadable.
            # Fall back to 0 if the ROM is not available.
            max_score = 0
    except ImportError:
        pass

    return JerichoGameData(
        display_name=game_name.replace("_", " ").title(),
        genre="interactive fiction",
        premise="A classic text adventure game. Explore, solve puzzles, and maximize your score.",
        max_score=max_score,
        scoring_description="Points from exploration, puzzle-solving, and collecting key items.",
    )
