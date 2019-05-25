import tensorflow as tf
import numpy as np
import operator
import collections
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import train
import pickle
import argparse
from numpy import linalg as LA




def prune_filter_weight(meta_graph, latest_checkpoint, num_filters):

    indices_List = []
    graph = tf.Graph()
    with graph.as_default():

        with tf.Session() as sess:
            imported_meta = tf.train.import_meta_graph(meta_graph)
            imported_meta.restore(sess, tf.train.latest_checkpoint(latest_checkpoint))

            variables_graph_nodes = [v.name for v in tf.trainable_variables() if (v.name.endswith('W:0') & (v.name.find('conv-maxpool')!=-1))]
            variables_names_bias = [v.name for v in tf.trainable_variables() if (v.name.endswith('b:0') & (v.name.find('conv-maxpool')!=-1))]
            variables_graph_nodes_1 = [v for v in tf.trainable_variables() if (v.name.find('conv-maxpool')!=-1)]
            variables_graph_nodes_2 = [v for v in tf.trainable_variables() if (v.name.find('conv-maxpool')!=-1) or (v.name.find('embedding')!=-1)]
            print(variables_graph_nodes_2)
            embed = sess.run('embedding/W:0')
            print('embed_shape',embed.shape)
            embeddings = embed[:,:-30]
            print(embeddings.shape)
            new_node = tf.get_default_graph().get_tensor_by_name('embedding/W:0') 
            change_shape_op = tf.assign(new_node, embeddings, validate_shape = False)
            print(change_shape_op.shape)
            sess.run(change_shape_op)

            def pruned_weights(variable_name):
                var = sess.run(variable_name)
              
                weight_var = []
        
                for i in range(num_filters):
                    weight_variance = np.var(var[:,:,:,i])
                    weight_var.append(weight_variance)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                weight_var_numpy = np.asarray(weight_var)
                weight_var_sorted=(weight_var_numpy.argsort())
                filters_to_prune = int(0.3*num_filters)
                #filters_to_prune = int(0.3*num_filters)
                print('filters_to_prune',filters_to_prune)
                indices = weight_var_sorted[:filters_to_prune]
                filters_retained = num_filters - filters_to_prune
                print('filters_retained',filters_retained)
                indices_retained = weight_var_sorted[-filters_retained:]
                print(indices)
                indices_List.append(indices_retained)
                #print('rem',var.shape)
                return var,indices_List,filters_retained

            for i,variable_name in enumerate(variables_graph_nodes):
                pruned_filter_weights, indices_List,filters_retained = pruned_weights(variable_name)
                #var = sess.run(variable_name)
                bias = sess.run(variables_names_bias[i])
                rem = np.empty([pruned_filter_weights.shape[0],embeddings.shape[1],1,filters_retained])
                b = np.empty([filters_retained])
                for j in range(len(indices_List[i])):
                    np.append(rem,pruned_filter_weights[:,:,:,indices_List[i][j]])
                    np.append(b,bias[[indices_List[i][j]]])
                new_node = tf.get_default_graph().get_tensor_by_name(variable_name)
                new_node_bias = tf.get_default_graph().get_tensor_by_name(variables_names_bias[i])
          
                change_shape_op = tf.assign(new_node, rem, validate_shape=False)
                change_shape_op_bias = tf.assign(new_node_bias, b, validate_shape=False)
                print(change_shape_op.shape[3])
                sess.run(change_shape_op)

                print(change_shape_op_bias.shape)
                sess.run(change_shape_op_bias)


            dir = os.path.dirname(os.path.realpath(__file__))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "pruned_data_by_l2norm_tobacco_1", "checkpoints"))
            checkpoint_prefix = os.path.join(out_dir, "model")

            print(sess.run(tf.get_default_graph().get_tensor_by_name('conv-maxpool-3/W:0')).shape) 
            
            if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                   
            
            #print(variables_graph_nodes_2)    
            variables_dict = {var.name.split(':')[0]: var for var in variables_graph_nodes_2}
            #print(variables_dict)
            saver = tf.train.Saver(variables_dict)
            path = saver.save(sess, checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(path))
            return filters_retained
        #print(tf.trainable_variables())


def make_graph(x_train, y_train, vocab_processor, x_dev, y_dev):
    

    cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=98,
                    filter_sizes=[3,4,5],
                    num_filters= 90,
                    l2_reg_lambda=0.0)


    with tf.Session() as sess:
        graph = tf.get_default_graph()
        print(tf.trainable_variables())

        #sess.run(tf.global_variables_initializer())
        #print(sess.run(tf.get_default_graph().get_tensor_by_name('output/W:0')).shape)
        restore_variables =  {v.name.split(':')[0]:v for v in tf.trainable_variables() if (v.name.find('conv-maxpool')!=-1) or (v.name.find('embedding')!=-1)}
        #restore_variables = {v.name.split(':')[0]:v for v in tf.trainable_variables() if (v.name.find('conv-maxpool')!=-1) or (v.name.find('embedding')!=-1)}
        print(restore_variables)
        saver = tf.train.Saver(restore_variables, max_to_keep=5)

        saver.restore(sess, tf.train.latest_checkpoint('./pruned_data_by_l2norm_tobacco_1/checkpoints/'))
       # print([n.name for n in tf.get_default_graph().as_graph_def().node])


        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        tf.add_to_collection("optimizer", train_op)
        sess.run(tf.variables_initializer(optimizer.variables()))

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "pruned_by_l2_norm", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(restore_variables, max_to_keep=5)    

        
        #sess.run(tf.variables_initializer(optimizer.variables()))
        sess.run(tf.global_variables_initializer())
        


        def train_step(x_batch, y_batch):
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 0.25
            }
            _, step, summaries,  loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            

        def dev_step(x_batch, y_batch, writer=None):

            feed_dict = {
             cnn.input_x: x_batch,
             cnn.input_y: y_batch, 
             cnn.dropout_keep_prob: 1.0
             }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op , cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            print("{}: step {}, loss {:g}, acc {:g},".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

                # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), 64, 10)
                # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev,writer=dev_summary_writer)
                print("")

                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--make_graph", dest="makegraph", action="store_true")
    args = parser.parse_args()
    return args                


if __name__ == '__main__':
    args = get_args()
   
    if args.prune:
        filters = prune_filter_weight("./runs/1557076524/checkpoints/model-21300.meta", './runs/1557076524/checkpoints/',128)
        #filters = prune_filter_weight("./pruned_by_l2_norm/1557075922/checkpoints/model-1000.meta", './pruned_by_l2_norm/1557075922/checkpoints',90)
        print(filters)
        
    elif args.makegraph:
        x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test = train.preprocess()
        make_graph(x_train, y_train, vocab_processor, x_dev, y_dev)




