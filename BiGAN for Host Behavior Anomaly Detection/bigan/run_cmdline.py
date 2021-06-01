import os
import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# network file
import bigan.cmdline_utilities as GAN
# data preprocess file
import data.cmdline as data

RANDOM_SEED = 13
# print frequency image tensorboard [20]
FREQ_PRINT = 20
# parameter of WGAN-GP
LAMBDA = 10

# to update neural net with moving avg variables, suitable for ss learning cf Saliman
def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def print_loss(g_loss, e_loss, d_loss, loss, nb_epochs):
    epoch = np.arange(1, nb_epochs + 1)
    plt.plot(epoch, g_loss)
    plt.plot(epoch, e_loss)
    plt.plot(epoch, d_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['g_loss', 'e_loss', 'd_loss'], loc='upper right')
    plt.savefig('./img/' + loss + '_' + str(nb_epochs) + '.jpg')
    #plt.show()
    plt.close()


# display some parameters
def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)


# visualize the train progress
def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


# save training logs, weights, biases, etc.
def create_logdir(method, weight, rd):
    return "bigan/train_logs/cmdline/{}/{}/{}".format(weight, method, rd)


def train_and_test(nb_epochs, weight, method, degree, dataset, random_seed, p_lr, loss_fun, p = False):
    """
    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
    """
    logger = logging.getLogger("BiGAN.train.{}.{}".format(dataset, method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=(None, 100), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train()
    trainx_copy = trainx.copy()

    # can be delete----
    testx, testy, contaminate_rate = data.get_test()

    lx = len(testx)
    ly = len(testy)

    # Parameters
    #starting_lr = GAN.learning_rate
    starting_lr = p_lr
    batch_size = GAN.batch_size
    latent_dim = GAN.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    # ==================================================================================
    logger.info('Building training graph...')

    logger.warning("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = GAN.generator
    enc = GAN.encoder
    dis = GAN.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    # get noise matrix[batch_size x latent_dim] from N(0,1) to generator
    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, latent_dim], mean=0, stddev=1)
        #z = tf.clip_by_value(z, -1, 1)
        x_gen = gen(z, is_training=is_training_pl)
        #z_gen_z = z_gen + z
        #x_gen = gen(z_gen_z, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        l_generator, inter_layer_rct = dis(z, x_gen, is_training=is_training_pl, reuse=True)
        #l_generator, inter_layer_ge = dis(z_gen_z, x_gen, is_training=is_training_pl, reuse=True)


    with tf.name_scope('loss_functions'):
        if loss_fun == 'crosse':
            # cross entropy loss
            # discriminator
            loss_dis_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder))
            loss_dis_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator))
            loss_discriminator = loss_dis_gen + loss_dis_enc
            # generator
            loss_generator = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator), logits=l_generator))
            # encoder
            loss_encoder = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder), logits=l_encoder))

        elif loss_fun == 'w':
            # w loss from https://zhuanlan.zhihu.com/p/25071913
            # discriminator
            loss_dis_enc = tf.reduce_mean(l_encoder)
            loss_dis_gen = tf.reduce_mean(l_generator)
            loss_discriminator = loss_dis_gen - loss_dis_enc
            # generator
            loss_generator = -tf.reduce_mean(l_generator)
            # encoder
            loss_encoder = tf.reduce_mean(l_encoder)


        elif loss_fun == 'wgp':
            # w loss with gp from https://blog.csdn.net/shanlepu6038/article/details/86539216
            # discriminator
            loss_dis_enc = tf.reduce_mean(l_encoder)
            loss_dis_gen = tf.reduce_mean(l_generator)
            loss_discriminator = loss_dis_gen - loss_dis_enc
            # generator
            loss_generator = -tf.reduce_mean(l_generator)
            # encoder
            loss_encoder = tf.reduce_mean(l_encoder)
            # GP
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            interpolates = alpha * x_gen + (1 - alpha) * input_pl
            temp, layer = dis(z, interpolates, is_training=is_training_pl, reuse=None)
            gradients = tf.gradients(layer, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            loss_discriminator += LAMBDA * gradient_penalty

        elif loss_fun == 'ls':
            # Least Squares loss from https://www.pianshen.com/article/9260508296/  a = 1 b = -1 c = 0
            # http://www.twistedwg.com/2018/10/05/GAN_loss_summary.html
            # For generator as real as possible
            l_generator = tf.nn.sigmoid(l_generator)
            l_encoder = tf.nn.sigmoid(l_encoder)
            # generator
            loss_generator = 0.5 * tf.reduce_mean(tf.square(l_generator))
            # encoder
            loss_encoder = 0.5 * tf.reduce_mean(tf.square(l_encoder))
            # discriminator
            loss_dis_enc = loss_encoder
            loss_dis_gen = loss_generator
            loss_discriminator = 0.5 * tf.reduce_mean(tf.squared_difference(l_encoder, 1.0)) + 0.5 * tf.reduce_mean(tf.squared_difference(l_generator, -1.0))

        elif loss_fun == 'hinge':
            # Hinge loss from http://www.twistedwg.com/2018/10/05/GAN_loss_summary.html
            # https://blog.csdn.net/u010946556/article/details/89487305
            # https://zhuanlan.zhihu.com/p/72195907
            l_encoder = tf.nn.tanh(l_encoder)
            l_generator = tf.nn.tanh(l_generator)
            # discriminator
            loss_dis_enc = tf.reduce_mean(tf.nn.relu(1.0 - l_encoder))
            loss_dis_gen = tf.reduce_mean(tf.nn.relu(1.0 + l_generator))
            loss_discriminator = loss_dis_gen + loss_dis_enc
            # generator
            loss_generator = -tf.reduce_mean(l_generator)
            # encoder
            loss_encoder = tf.reduce_mean(l_encoder)

        else:
            logger.warning("Loss function error\n")
            sys.exit(0)


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]
        dvars = [var for var in tvars if 'discriminator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        if loss_fun not in ['w', 'wgp']:
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
            optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
            clip_disc_weights = None
            disc_iters = 1

        elif loss_fun == 'w':
            optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='gen_optimizer')
            optimizer_enc = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='enc_optimizer')
            optimizer_dis = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='dis_optimizer')
            disc_iters = 1
            clip_ops = []
            # weight clipping[-0.01,0.01]
            for var in tvars:
                if var.name.startswith("discriminator"):
                    clip_bounds = [-0.01, 0.01]
                    clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif loss_fun == 'wgp':
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
            optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
            disc_iters = 1
            clip_disc_weights = None

        else:
            logger.warning("Optimizer Error")
            sys.exit(0)

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    # ==================================================================================
    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                                keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator_ema), logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1, keep_dims=False, name='d_loss')

            else:
                logger.warning("Discriminator Loss Error")
                sys.exit(0)
            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score

    logdir = create_logdir(weight, method, random_seed)

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None, save_model_secs=120)

    logger.info('Start training...')
    with sv.managed_session() as sess:

        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0
        d_loss = []
        g_loss = []
        e_loss = []
        while not sv.should_stop() and epoch < nb_epochs:
            lr = starting_lr
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # train discriminator
                for i in range(disc_iters):
                    feed_dict = {input_pl: trainx[ran_from:ran_to],
                                 is_training_pl: True,
                                 learning_rate: lr}
                    _, ld, sm = sess.run([train_dis_op,
                                          loss_discriminator,
                                          sum_op_dis], feed_dict=feed_dict)
                    train_loss_dis += ld
                    writer.add_summary(sm, train_batch)
                    if clip_disc_weights is not None:
                        _ = sess.run(clip_disc_weights)

                # train generator and encoder
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate: lr}
                _, _, le, lg, sm = sess.run([train_gen_op,
                                             train_enc_op,
                                             loss_encoder,
                                             loss_generator,
                                             sum_op_gen],
                                            feed_dict=feed_dict)

                train_loss_gen += lg
                train_loss_enc += le
                writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train

            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))

            d_loss.append(train_loss_dis)
            g_loss.append(train_loss_gen)
            e_loss.append(train_loss_enc)

            epoch += 1

        if p:
            print_loss(g_loss, e_loss, d_loss, loss_fun, nb_epochs)

        logger.warning('Testing evaluation...')
        # shuffling  dataset
        inds = rng.permutation(testx.shape[0])
        testx = testx[inds]
        testy = testy[inds]
        scores = []
        inference_time = []

        # Create scores
        for t in range(nr_batches_test):
            # construct randomly permuted minibatches
            ran_from = t * batch_size
            ran_to = (t + 1) * batch_size
            begin_val_batch = time.time()

            feed_dict = {input_pl: testx[ran_from:ran_to],
                         is_training_pl: False}

            scores += sess.run(list_scores, feed_dict=feed_dict).tolist()
            inference_time.append(time.time() - begin_val_batch)

        logger.info('Testing : mean inference time is %.4f' % (np.mean(inference_time)))

        ran_from = nr_batches_test * batch_size
        ran_to = (nr_batches_test + 1) * batch_size
        size = testx[ran_from:ran_to].shape[0]
        # 全1矩阵填充剩余部分
        fill = np.ones([batch_size - size, 100])
        batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        feed_dict = {input_pl: batch, is_training_pl: False}

        batch_score = sess.run(list_scores, feed_dict=feed_dict).tolist()

        scores += batch_score[:size]

        # Contaminate rate = x % ===> highest x % are anomalous
        per = np.percentile(scores, 100 - contaminate_rate * 100)

        y_pred = scores.copy()
        y_pred = np.array(y_pred)
        y_scores = scores.copy()
        y_scores = np.array(y_scores)

        inds = (y_pred < per)
        inds_comp = (y_pred >= per)
        cc = 0
        ccc = 0
        for i in inds:
            if i == True:
                cc += 1
        for i in inds_comp:
            if i == True:
                ccc += 1

        y_pred[inds] = 0
        y_pred[inds_comp] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(testy, y_pred, average='binary')
        accuracy = accuracy_score(testy, y_pred)

        print("Initialization: len(testx):{} len(testy):{} contaminate_rate:{}".format(lx, ly, contaminate_rate))
        print("After discriminate: 0-number:{} 1-number:{}".format(cc, ccc))
        print("Testing : Acc = %.4f | Prec = %.4f | Rec = %.4f | F1 = %.4f " % (accuracy, precision, recall, f1))
        with open('result.txt', 'a') as fin:
            fin.write("lossfunction:" + loss_fun + ' L-norm:' + str(degree) + '\n')
            fin.write("contaminate_rate_score:{}\n".format(per))
            fin.write("learningrate:" + str(p_lr))
            fin.write(" batch" + str(batch_size))
            fin.write(" epochs:" + str(nb_epochs))
            fin.write(' acc:' + str(accuracy) + ' prec:' + str(precision) + ' rec:' + str(recall) + ' f1:' + str(f1) + '\n')
        fin.close()


# training process
def train(nb_epochs, weight, method, degree, dataset, random_seed=24, p_lr = 0.00003, loss = 'crosse', plot = 'n'):
    if plot in ['y','Y']:
        p = True
    else:
        p = False
    d = degree
    # Set degree for every loss functions
    if loss == 'crosse':
        d = 4
    elif loss == 'w' or loss == 'ls':
        d = 3
    elif loss == 'wgp':
        d = 1
    elif loss == 'hinge':
        d = 9
    else:
        logging.warning("Wrong Loss function")
        sys.exit(0)
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, d, dataset, random_seed, p_lr, loss, p = p)
