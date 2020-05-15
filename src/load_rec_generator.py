import argparse
from recommendation.recommender_utils.Solver import Solver
from time import time
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path')
    parser.add_argument('--experiment_name', nargs='?', default='free_adv_original',
                        help='original_images, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--model', nargs='?', default='VBPR',
                        help='recommender models: VBPR')
    parser.add_argument('--emb1_K', type=int, default=64, help='size of embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', nargs='?', default='[0.01,1e-4,1e-3]', help='learning rate')
    parser.add_argument('--verbose', type=int, default=1000, help='verbose')
    parser.add_argument('--epoch', type=int, default=4000, help='epochs')
    parser.add_argument('--regs', nargs='?', default='[1e-1,1e-3,0]', help='lambdas for regularization')
    parser.add_argument('--lmd', type=float, default=0.1,
                        help='lambda for balance the common loss and adversarial loss')
    parser.add_argument('--keep_prob', type=float, default=0.6, help='keep probability of dropout layers')
    parser.add_argument('--adv', type=int, default=0, help='adversarial training')
    parser.add_argument('--adv_type', nargs='?', default='grad', help='adversarial training type: grad, rand')
    parser.add_argument('--cnn', nargs='?', default='resnet', help='cnn type: resnet50')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon for adversarial')
    parser.add_argument('--weight_dir', nargs='?', default='rec_model_weights', help='directory to store the weights')
    parser.add_argument('--result_dir', nargs='?', default='rec_results', help='directory to store the predictions')

    parser.add_argument('--topk', type=int, default=150,
                        help='top k predictions to store before the evaluation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('Device gpu: {0}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    solver = Solver(args)
    print(args)
    if args.adv == 0:
        # I need to duplicate because it divide by two in the code
        solver.epoch = solver.epoch*2
        solver.load()

    start_time = time()

    print('START Training of the Recommender Model at {0}.'.format(start_time))
    solver.new_store_predictions(4000)
    print('END Training of the Recommender Model in {0} secs.'.format(time() - start_time))

