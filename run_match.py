from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager

CURRENT_FILE = Path(__file__).parent

MATCH_CONFIG_PATH = CURRENT_FILE / "minimal.toml"
RLBOT_SERVER_FOLDER = Path("C:\\Users\\ddthj\\Documents\\Repositories\\RLBotCS\\RLBotCS\\bin\\Release\\net8.0")


if __name__ == "__main__":
    match_manager = MatchManager(RLBOT_SERVER_FOLDER)

    match_manager.ensure_server_started()
    match_manager.start_match(MATCH_CONFIG_PATH)

    sleep(5)

    while match_manager.game_state != flat.GameStateType.Ended:
        sleep(0.1)

    match_manager.shut_down()
