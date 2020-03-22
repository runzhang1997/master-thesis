import tensorflow as tf
import numpy as np
import argparse
import network
import os
import sys
import time
import horovod.tensorflow as hvd
#from tensorflow.python.client import timeline

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--patch_size', help='.',type=int)
    parser.add_argument('--epochs', help='.',type=int)
    parser.add_argument('--patch_1', help='.',type=int)
    parser.add_argument('--patch_2', help='.',type=int)
    parser.add_argument('--patch_3', help='.',type=int)
    parser.add_argument('--shuffel', help='.',type=int)
    parser.add_argument('--kd', default=1, help='.', type=int)
    parser.add_argument('--batch_size', help='.',type=int)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--discrimator_learning_rate', type=float, default=6e-4, help='Learning rate for Adam.')
    parser.add_argument('--adv_weight', type=float, default=5e-3, help='adversarial loss')
    parser.add_argument('--train_data', help='Path of train data.')  
    parser.add_argument('--load', action='store_true', default=False, help='the version of network' )
    parser.add_argument('--checkpoints', help='checkpoints' )
    parser.add_argument('--gan_type', help='checkpoints' )
    parser.add_argument('--pretrain_generator', action='store_true', help='.')
    parser.add_argument('--norm', action='store_true', help='instance normlization or not')
    parser.add_argument('--relu', action='store_true', help='instance normlization or not')
    parser.add_argument('--contentloss', default="L1", help='checkpoints' )
    parser.add_argument('--validation_data', help='Path of valid data.')  
    parser.add_argument('--valid',action='store_true')
    parser.add_argument('--run_60', action='store_true', help='Default 20->10m.')
    parser.add_argument('--network_data', help='location to save the intermediate model' ) 
    args = parser.parse_args()
    
    def decode(serialized_example):
        """decode the serialized example"""   
        patch_1 = args.patch_1
        patch_2 = args.patch_2
        
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'train_input1': tf.FixedLenFeature([], tf.string),
                                           'train_input2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
                                       
        train_input_10 = tf.decode_raw(features['train_input1'], tf.float32)
        train_input_10 = tf.reshape(train_input_10, [patch_1, patch_1, 4])
        train_input_20 = tf.decode_raw(features['train_input2'], tf.float32)
        train_input_20 = tf.reshape(train_input_20, [patch_2, patch_2, 6])
        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, [patch_1, patch_1, 6])
        
        return train_input_10, train_input_20, label
        
    def decode60(serialized_example):
        """decode the serialized example"""   
        patch_1 = args.patch_1
        patch_2 = args.patch_2
        patch_3 = args.patch_3
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'train_input1': tf.FixedLenFeature([], tf.string),
                                           'train_input2': tf.FixedLenFeature([], tf.string),
                                           'train_input3': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
                                       
        train_input_10 = tf.decode_raw(features['train_input1'], tf.float32)
        train_input_10 = tf.reshape(train_input_10, [patch_1, patch_1, 4])
        train_input_20 = tf.decode_raw(features['train_input2'], tf.float32)
        train_input_20 = tf.reshape(train_input_20, [patch_2, patch_2, 6])
        train_input_60 = tf.decode_raw(features['train_input3'], tf.float32)
        train_input_60 = tf.reshape(train_input_60, [patch_3, patch_3, 2])
        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, [patch_1, patch_1, 2])   
        return train_input_10, train_input_20, train_input_60, label
        
    hvd.init()
    run_config = tf.ConfigProto(use_per_session_threads=True)
    run_config.gpu_options.allow_growth=True
    run_config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    if args.run_60:
        out_path = 'network_data/run_60_%s/'%args.network_data
        if not os.path.isdir(out_path):
            if hvd.rank() ==0:
                os.mkdir(out_path)
    else:
        out_path = 'network_data/run_20_%s/'% args.network_data
        if not os.path.isdir(out_path):
            if hvd.rank() ==0:
                os.mkdir(out_path)    

    tf.reset_default_graph() 
      
    dataset = tf.data.TFRecordDataset(args.train_data)
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.repeat(args.epochs)  
    #dataset = dataset.cache()
    dataset = dataset.shuffle(args.shuffel) # shuffle
    if args.run_60:
        dataset = dataset.map(decode60)
    else:
        dataset = dataset.map(decode)
    #dataset = dataset.apply(tf.contrib.data.map_and_batch(decode, batch_size, drop_remainder=True))
    dataset = dataset.batch(args.batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    if args.run_60:
        train10, train20, train60, label = iterator.get_next()
    else:
        train10, train20, label = iterator.get_next()
           
    # set up learning rate.
    '''
    global_step_generator = tf.Variable(0, trainable=False)
    global_step_discriminator = tf.Variable(0, trainable=False)
    lr_discriminator = tf.train.exponential_decay(args.discrimator_learning_rate, 
        global_step_discriminator, 8e4, 0.5, staircase=False)
    lr_generator = tf.train.exponential_decay(args.generator_learning_rate, 
        global_step_generator, 4e4, 0.5, staircase=False)  
    '''
    # Set up models
    discriminator = network.Discriminator(loss_type = args.gan_type, image_size=args.patch_size, 
        batch_size = args.batch_size, norm=args.norm, run_60=args.run_60)
    generator = network.Generator(adversarial_loss = args.gan_type, content_loss= args.contentloss, 
        batch_size = args.batch_size,discriminator=discriminator, norm=False, 
        adv_weight=args.adv_weight, relu=args.relu, run_60=args.run_60)
        
    # Generator
    if args.run_60:
        g_y_pred = generator.forward(train10, train20, train60)
    else:
        g_y_pred = generator.forward(train10, train20)
        
    contloss = generator._content_loss(label, g_y_pred) 
    adverloss = generator._adversarial_loss(label, g_y_pred)
    g_loss = contloss + adverloss    
    args.generator_learning_rate
    #g_train_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_generator, beta1=0., beta2=0.9),  
    #compression= hvd.Compression.fp16).minimize(g_loss, global_step=global_step_generator, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
    g_train_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(args.generator_learning_rate, beta1=0., beta2=0.9), 
    compression= hvd.Compression.fp16).minimize(g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        
    # pretrain generator
    pretrain_loss = generator.pretrain_loss(label, g_y_pred)
    pretrain_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(args.generator_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08),
    compression= hvd.Compression.fp16).minimize(pretrain_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
    #pretrain_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_generator, beta1=0.9, beta2=0.999, epsilon=1e-08),
    #compression= hvd.Compression.fp16).minimize(pretrain_loss, global_step=global_step_generator, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
           
    # Discriminator
    d_y_real_pred_logits = discriminator.forward(label)
    d_y_fake_pred_logits = discriminator.forward(g_y_pred)
    d_loss = discriminator.loss(d_y_real_pred_logits, d_y_fake_pred_logits)
    if args.gan_type == 'wgan-gp':
        gp = discriminator.gradient_penalty(g_y_pred, label)
        d_loss = gp + d_loss
    #args.discrimator_learning_rate
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    with tf.control_dependencies(update_ops):   
        #d_train_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(lr_discriminator, beta1=0., beta2=0.9), 
        #compression= hvd.Compression.fp16).minimize(d_loss, global_step=global_step_discriminator, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
        d_train_step = hvd.DistributedOptimizer(tf.train.AdamOptimizer(args.discrimator_learning_rate, beta1=0., beta2=0.9), 
        compression= hvd.Compression.fp16).minimize(d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) 
    #run_metadata = tf.RunMetadata()
    
    if args.valid == True:
        validation = tf.data.TFRecordDataset(args.validation_data)
        validation = validation.shard(hvd.size(), hvd.rank())
        if args.run_60:
            validation = validation.map(decode60)
        else:
            validation = validation.map(decode)
        validation = validation.batch(32, drop_remainder=True)
        validation = validation.prefetch(buffer_size=1)
        iterator_validation = validation.make_initializable_iterator()
        if args.run_60:
            valid10, valid20, valid60, valid_label = iterator_validation.get_next()
        else:
            valid10, valid20, valid_label = iterator_validation.get_next()
        if args.run_60:
            valid_g_y_pred = generator.forward(valid10, valid20, valid60)
        else:
            valid_g_y_pred = generator.forward(valid10, valid20)
        valid_loss = generator.valid_loss(valid_label, valid_g_y_pred)    
        valid_loss_best = 100
    
    if args.pretrain_generator:
        print ("pretrain generator!!")
        with tf.Session(config=run_config) as sess:       
            # Initialize
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())       
            if hvd.rank() ==0:
                writer = tf.summary.FileWriter(out_path + 'graphs', sess.graph)
            e = 0
            saver = tf.train.Saver(max_to_keep=24)
            iteration = 0
            if args.load:                
                latest_checkpt = tf.train.latest_checkpoint(args.checkpoints)
                e = int(os.path.basename(latest_checkpt).split('-')[1])
                iteration = e   
                saver.restore(sess, latest_checkpt)
                print ('load success!')
            print('Training...')        
            hvd.broadcast_global_variables(0).run()
            print('Training starts...')                     
            global_start = time.time()
            #content_l_ =[]
            try:
                while True:
                    iteration += 1
                    
                    #if iteration == 100 and hvd.rank() == 0:
                    #    print (time.ctime(time.time()))
                    
                   
                    #if iteration == 300 and hvd.rank() == 0:
                    #    print (time.ctime(time.time())) 
                    #    break
                    
                    #pretrain_loss_, _ = sess.run([pretrain_loss, pretrain_step])
                    pretrain_loss_, lr_step, _, gen_step = sess.run([pretrain_loss, lr_generator, pretrain_step, global_step_generator])
                    # validation step
                    if iteration % 20000 ==0 and args.valid==True:
                        sess.run(iterator_validation.initializer)
                        batch_valid_count = int(52441*5/hvd.size()/229)
                        #batch_valid_count = int(10140/hvd.size()/32)
                        loss_valid_grounp = []
                        for i in range(batch_valid_count):
                            valid_loss_batch = sess.run(valid_loss)
                            loss_valid_grounp.append(valid_loss_batch)                        
                        loss_iteration = np.mean(loss_valid_grounp)
                        loss_iteration=tf.convert_to_tensor(loss_iteration, dtype=np.float32)
                        loss_average_iteration = hvd.allreduce(loss_iteration)
                        loss_average_iteration = sess.run(loss_average_iteration)
                        #print(type(loss_average_iteration))
                        loss_average_iteration=np.mean(loss_average_iteration)
                        #print(type(loss_average_iteration))
                        if hvd.rank() == 0:
                            summary = tf.Summary()
                            summary.value.add(tag='validloss', simple_value=loss_average_iteration)
                            writer.add_summary(summary, iteration)
                        if loss_average_iteration < valid_loss_best:
                            valid_loss_best = loss_average_iteration
                            if hvd.rank() == 0:
                                saver.save(sess, os.path.join(out_path, 'pretrain'), global_step=iteration, write_meta_graph=False)
                    if hvd.rank() == 0:
                        #print (np.mean(content_l_))
                        summary = tf.Summary()
                        summary.value.add(tag='learning_rate', simple_value=np.mean(lr_step))
                        summary.value.add(tag='content loss', simple_value=pretrain_loss_)
                        writer.add_summary(summary, iteration)
                        #print(pretrain_loss_)
                        #    #wandb.log({'content loss': np.mean(content_l_)},step=e)
                        #    #print (np.mean(content_l_))
                        #iteration = 0
                    end = time.time()  
                    if end - global_start > 3600 and hvd.rank() == 0:
                        global_start = time.time()
                        saver.save(sess, os.path.join(out_path, 'pretrain'), global_step=iteration, write_meta_graph=False)
                    if iteration == 400000 and hvd.rank() == 0:
                        saver.save(sess, os.path.join(out_path, 'pretrain'), global_step=iteration, write_meta_graph=False)
            except tf.errors.OutOfRangeError:
                print ('ERROR!')
                pass
    else:
        with tf.Session(config=run_config) as sess:
            # Initialize
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())         
            if hvd.rank() ==0:
                writer = tf.summary.FileWriter(out_path + 'graphs', sess.graph)               
            e = 0
            iteration = 0
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
            if args.load:
                latest_checkpt = tf.train.latest_checkpoint(args.checkpoints)
                #e = int(os.path.basename(latest_checkpt).split('-')[1])
                #iteration = e                  
                saver.restore(sess, latest_checkpt)
                print ('load success!')
            print('Training..')
            hvd.broadcast_global_variables(0).run()
            #sess.run(hvd.broadcast_global_variables(0))
            print('Training starts...')
            '''
            total_parameters = 0
            for variable in tf.trainable_variables('generator'):
            # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print(total_parameters)
            total_parameters = 0
            for variable in tf.trainable_variables('discriminator'):
            # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print(total_parameters)
            '''
            global_start = time.time()
            batch_d_l = []
            content_l_ =[]
            adv_l_ = []
            batch_g_l = []
            gradeint_average = []
            try:
                while True:
                    #break
                    iteration += 1
                    for k in range(args.kd):
                        sess.run(d_train_step)
                    batch_d_loss, _ = sess.run([d_loss, d_train_step])
                    #batch_d_l_.append(batch_d_loss_)
                    #gradient_.append(gradient_p)
                    batch_d_l.append(batch_d_loss)
                    content_l, adv_l, batch_g_loss, _ = sess.run([contloss, adverloss, g_loss, g_train_step])
                    content_l_.append(content_l)
                    adv_l_.append(adv_l)
                    batch_g_l.append(batch_g_loss)
                    #gradeint_average.append(batch_gp)
                    if hvd.rank() == 0:
                        summary = tf.Summary()
                        #summary.value.add(tag='perceptron loss', simple_value=np.mean(gradeint_average))
                        #summary.value.add(tag='gradient_p', simple_value=np.mean(gradient_))
                        summary.value.add(tag='batch_d_loss', simple_value=np.mean(batch_d_l))
                        summary.value.add(tag='content loss', simple_value=np.mean(content_l_))
                        summary.value.add(tag='adversarial loss', simple_value=np.mean(adv_l_))
                        summary.value.add(tag='batch_g_loss', simple_value=np.mean(batch_g_l))
                        writer.add_summary(summary, iteration)         
                    end = time.time()
                    if end - global_start > 7100 and hvd.rank() == 0:
                        global_start = time.time()
                        saver.save(sess, os.path.join(out_path, 'weights'), global_step=iteration, write_meta_graph=False)
                    #Train discriminator
                    '''    
                    if e == 5 and iteration == 20:
                        batch_d_loss, _ = sess.run([d_loss, d_train_step], options=run_options, run_metadata=run_metadata)
                        epoch_disc_loss.append(batch_d_loss)
                    
                        if hvd.rank() ==0:
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open('timeline_discriminator_hvd.json', 'w') as f:
                                f.write(chrome_trace)
                
                        batch_g_loss, _ = sess.run([g_loss,g_train_step],options=run_options,run_metadata=run_metadata)
                        epoch_gen_loss.append(batch_g_loss)
                    
                        if hvd.rank()==0:
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open('timeline_genrator_hvd.json', 'w') as f:
                                f.write(chrome_trace)                            
                        iteration += 1
                    ''' 
            except tf.errors.OutOfRangeError:
                print ('ERROR!')
                pass
