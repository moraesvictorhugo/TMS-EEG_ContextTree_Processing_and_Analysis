# All relevant constants separated by files name

# trigger constants
# START_SEQUENCE = False # Button in the future
# DEBUG_ARDUINO = False
# STIMULI_NUMBER = 50 #Number of total stimuli
ITI = [1,3] #  inter-trial interval (ITI), have in mind that when using MagicPy, it adds a 3 seconds delay due to sleep need
create_navigation_marker = False

# Publisher messages from invesalius
PUB_MESSAGES = [
    'Coil at target',
    'Marker label',
]

# # txt config constants
# DIR_PATH = 'Markers-sequence'
# sequence_file_name = 'test.txt'

# # update_navigation constants