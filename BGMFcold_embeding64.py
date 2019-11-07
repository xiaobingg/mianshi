import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from BcoldDataset import Dataset
from Bcoldevaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import BXGMFcold1_3_embeding64


def parse_args():
    parser = argparse.ArgumentParser(description="Run BGMFcold_embeding64.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='Pretrain/ml-1m_BXGMFcold1_3_embeding64.h5',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')

    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  init=init_normal, W_regularizer=l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  init=init_normal, W_regularizer=l2(regs[1]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings
    predict_vector = merge([user_latent, item_latent], mode='mul')

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model


def load_pretrain_model(model, gmf_model):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('x_user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('x_item_embedding').get_weights()
    model.get_layer('user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('item_embedding').set_weights(gmf_item_embeddings)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('x_prediction').get_weights()
    new_weights = gmf_prediction[0]
    new_b = gmf_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for i in train:
        # print i
        # print i[0],i[1],i[2]
        user_input.append(i[0])
        item_input.append(i[1])
        labels.append(i[2])

        # positive instance

    return user_input, item_input, labels


if __name__ == '__main__':
    f = open("C:/Users/19678/Desktop/BGMFcold_embeding64", "w+")
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("BGMFcold_embeding64 arguments: %s" % (args))
    print >> f, "BGMFcold_embeding64 arguments: %s" % (args)
    # model_out_file = 'Pretrain/%s_GMF.h5' %(args.dataset)

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = 11163,5019
    print("Load data done [%.1f s]. #user=%d, #item=%d,#test=%d"
          % (time() - t1, num_users, num_items, len(testRatings)))
    print >> f, "Load data done [%.1f s]. #user=%d, #item=%d,#test=%d" % (
    time() - t1, num_users, num_items, len(testRatings))
    # Build model
    model = get_model(num_users, num_items, num_factors, regs)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    # print(model.summary())

    # Load pretrain model
    if mf_pretrain != '':
        gmf_model = BXGMFcold1_3_embeding64.get_model(num_users, num_items, num_factors)
        gmf_model.load_weights(mf_pretrain)
        model = load_pretrain_model(model, gmf_model)
        print("Load pretrained BXGMFcold1_3_embeding64(%s) models done. " % (mf_pretrain))
        print >> f, "Load pretrained BXGMFcold1_3_embeding64(%s) models done. " % (mf_pretrain)

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    # p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))
    print >> f, 'Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1)
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            print >> f, ('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                         % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                #     model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    print >> f, "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg)
    # if args.out > 0:
    #     print("The best GMF model is saved to %s" %(model_out_file))
    #     print >>f,"The best GMF model is saved to %s" %(model_out_file)