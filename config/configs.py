# OLD EXPERIMENTS
data_path = '../data/{0}/'
classes_path = '../data/imagenet_classes.txt'
training_path = data_path + 'trainingset.tsv'
test_path = data_path + 'testset.tsv'
original = data_path + 'original/'
images_path = original + 'images/'
output_classes_path = original + 'classes.csv'
features_path = original + 'features.npy'

# NEW EXPERIMENTS
new_data_path = '../new_data/{0}/'
new_original = new_data_path + 'original/'
features_DVBPR_path = new_original + 'features_DVBRP.npy'

# RESULTS
results_path = '../results/rec_results/{0}_top{1}_ep{2}_{3}.tsv'
metrics_path = '../results/rec_results/{0}_top{1}_ep{2}_{3}_metrics.tsv'
