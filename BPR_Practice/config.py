dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# paths
#main_path = '/home/share/guoyangyang/recommendation/NCF-Data/'
main_path = '/home/gyuseok/neural_collaborative_filtering/Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './Checkpoints/' # model save path
BPR_model_path = model_path + 'NeuMF.pth'