# coding:utf-8
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Bevaluate import evaluate_model
from BDataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
import BXMLP1_3_embeding32_16_8_4


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run BMLP_embeding32_16_8_4.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[32,16,8,4]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=3,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mlp_pretrain', nargs='?', default='Pretrain/ml-1m_BXMLP1_3_embeding32_16_8_4.h5',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')

    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    # print len(layers),len(reg_layers)
    assert len(layers) == len(reg_layers)

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='item_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))

    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')

    # MLP layers
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model


def load_pretrain_model(model, mlp_model, num_layers):
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('item_embedding').set_weights(mlp_item_embeddings)
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = mlp_prediction[0]
    new_b = mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


# 训练时正负样本比为1:4
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for i in train:
        # print i
        # print i[0],i[1],i[2]
        user_input.append(i[0])
        item_input.append(i[1])
        labels.append(i[2])
    return user_input, item_input, labels


if __name__ == '__main__':
    f = open("C:/Users/19678/Desktop/BMLP_embeding32_16_8_4", "w+")
    args = parse_args()
    path = args.path
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    mlp_pretrain = args.mlp_pretrain
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("BMLP_embeding32_16_8_4 arguments: %s " % (args))
    # model_out_file = 'Pretrain/%s_BMLPoptimal.h5' % (args.dataset)

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
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Load pretrain model
    if mlp_pretrain != '':
        mlp_model = BXMLP1_3_embeding32_16_8_4.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, mlp_model, len(layers))
        print("Load pretrained BXMLP1_3_embeding32_16_8_4 (%s) models done. " % (mlp_pretrain))
        print >> f, "Load pretrained BXMLP1_3_embeding32_16_8_4 (%s) models done. " % (mlp_pretrain)

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))
    print >> f, 'Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1)

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
            print >> f, 'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' % (
            epoch, t2 - t1, hr, ndcg, loss, time() - t2)
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                #     model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    print >> f, "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg)
    # if args.out > 0:
    #     print("The best MLP model is saved to %s" % (model_out_file))
