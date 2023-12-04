TRAIN_DIR = 'data/HT21/train/HT21-0'
DATA_FOLDER = '/img1/'
GT_FILE = '/gt/gt.txt'
# We split the data to actual train data and validation data that will be used during the training process.
# To ensure generalization, the entirety of the last scene was chosen since validation ratio of 997/5741 is close to industry standard
TRAIN_DATA_DIRS = [TRAIN_DIR + str(i) + DATA_FOLDER for i in range(1,4)]
VALIDATION_DATA_DIRS = [TRAIN_DIR + '4' + DATA_FOLDER]
TRAIN_GT_FILES = [TRAIN_DIR + str(i) + GT_FILE for i in range(1,4)]
VALIDATION_GT_FILES = [TRAIN_DIR + '4' + GT_FILE]

## NOT_USED ##
TEST_DIR = 'data/HT21/test/HT21-1'
DET_FILE = '/det/det.txt'
TRAIN_DET_FILES = [TRAIN_DIR + str(i) + DET_FILE for i in range(1,5)]
TEST_DATA_DIRS = [TEST_DIR + str(i) + DATA_FOLDER for i in range(1,6)]
TEST_DET_FILES = [TEST_DIR + str(i) + DET_FILE for i in range(1,5)]
##############