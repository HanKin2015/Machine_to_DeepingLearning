
TRAIN_BLACK_IMAGE_MATRIX_PATH = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\data\\train_black_image_matrix.csv'
TRAIN_WHITE_IMAGE_MATRIX_PATH = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\data\\train_white_image_matrix.csv'
TEST_IMAGE_MATRIX_PATH = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\data\\test_image_matrix.csv'

train_white_dataset = pd.read_csv(TRAIN_WHITE_IMAGE_MATRIX_PATH)
train_black_dataset = pd.read_csv(TRAIN_BLACK_IMAGE_MATRIX_PATH)
test_dataset        = pd.read_csv(TEST_IMAGE_MATRIX_PATH)
train_white_dataset.shape, train_black_dataset.shape, test_dataset.shape