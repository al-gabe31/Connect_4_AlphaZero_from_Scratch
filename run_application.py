from db_management import *
from game_window import *

database_location = 'db/Connect_4.db'
neural_network = retrieve_neural_network(
    database_location=database_location,
    neural_network_id=15 # we'll be using Alpha Horizons v2 since that seems to be the best version for now
)



if __name__ == '__main__':
    run_application(neural_network=neural_network)