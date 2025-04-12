def get_num_of_requirements(game_name):
    games = {
        'dice_game': 25,
        'arkanoid': 19,
        'snake': 14,
        'scopa': 16,
        'pong': 20
    }
    return games[game_name]