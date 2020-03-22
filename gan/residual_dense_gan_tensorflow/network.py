import tensorflow as tf

'''
def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape().as_list()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset
'''
'''
def InstanceNorm(input, axis=[2,3] , decay=0.999, name='InstanceNorm',trainable=True):   
    #axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]
    #shape=np.array(fdim)
    #shape[axis]=1
    
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim , dtype=tf.float32,initializer=tf.constant_initializer(value=0.0),trainable=trainable)
        gamma = tf.get_variable('gamma', fdim, dtype=tf.float32,initializer=tf.constant_initializer(value=1.0),trainable=trainable)
        
        instance_mean, instance_variance = tf.nn.moments(input, axis ,keep_dims=True)
    
    return tf.nn.batch_normalization(input, instance_mean, instance_variance, beta, gamma, 1e-3)
'''

class Generator:
    def __init__(self, discriminator, content_loss='L1', 
        batch_size=32, relu=False, 
        adversarial_loss='Vanila-gan', run_60=False, predict=False, 
        adv_weight=5e-3, norm=False):
        #self.learning_rate = learning_rate
        self.discriminator = discriminator
        self.adversarial_loss = adversarial_loss
        self.init_kernel = tf.initializers.he_normal(seed=111)
        self.run_60 = run_60
        self.predict=predict
        self.norm = norm
        self.content_loss=content_loss
        self.batch_size = batch_size
        self.relu = relu
        self.graph_created = False
        self.adv_weight=5e-3
        
    def dense_block(self, input):
        # pad the same or 1(sample implementation)
        x1 = tf.layers.conv2d(input, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
        x1 = tf.nn.leaky_relu(x1, alpha=.2)
        
        x2 = tf.concat([input, x1],axis=-1)
        x2 = tf.layers.conv2d(x2, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
        x2 = tf.nn.leaky_relu(x2, alpha=.2)
        
        x3 = tf.concat([input, x1, x2],axis=-1)
        x3 = tf.layers.conv2d(x3,filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
        x3 = tf.nn.leaky_relu(x3, alpha=.2)
        
        x4 = tf.concat([input, x1, x2, x3],axis=-1)
        x4 = tf.layers.conv2d(x4,filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
        x4 = tf.nn.leaky_relu(x4, alpha=.2)
        
        x5 = tf.concat([input, x1, x2, x3, x4],axis=-1)
        x5 = tf.layers.conv2d(x5, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
        x = x5 * 0.2 + input
        return x

    def RRDB(self, input):
        x = self.dense_block(input)
        x = self.dense_block(x)
        x = self.dense_block(x)
        out = x * 0.2 + input
        return out
    
    def _channel_attention(self, x, f, reduction):
        skip_conn = tf.identity(x)
        c = x.get_shape()[-1]
        x = tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

        x = tf.layers.conv2d(x, filters=f // reduction,padding='same', kernel_initializer=self.init_kernel, strides=1, kernel_size=1)
        if self.relu:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
        x = tf.layers.conv2d(x, filters=f, padding='same', kernel_initializer=self.init_kernel, strides=1, kernel_size=1)
        x = tf.nn.sigmoid(x)
        return tf.multiply(skip_conn, x)
        
    def _residual_channel_attention_Block(self, x, filters):
        """RCAB"""
        skip = tf.identity(x)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        # try to use instance normalization.
        if self.norm:
            x = tf.layers.batch_normalization(x, training=True)
        if self.relu:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        
        x= self._channel_attention(x, f=filters, reduction=16)
        #x = x * 0.1
        return x + skip   
    
    def _residual_channel_attention_block_project(self, x, filters):
        skip = tf.identity(x)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        # try to use instance normalization.
        if self.norm:
            x = tf.layers.batch_normalization(x, training=True)
            
        if self.relu:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        
        x= self._channel_attention(x, f=filters, reduction=16)
        #x = x * 0.1
        skip = tf.layers.conv2d(skip, kernel_size=1, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        return x + skip
        
    def _residual_group(self, x, filters):
        skip = tf.identity(x)
        x = self._residual_channel_attention_block_project(x, filters)
        for i in range(10):
            x = self._residual_channel_attention_Block(x, filters)
        return tf.concat([skip, x],-1)
    
    def _dense_residual_channel_attention(self, x, filters):
        for i in range(4):
            x = self._residual_group(x, filters)
        x_last = tf.layers.conv2d(x, kernel_size=1, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')       
        return x_last
        
    '''the Dsen2 residual block'''
    def _ResidualBlock(self, x, filters):
        """Residual block a la ResNet"""
        skip = tf.identity(x)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        # try to use instance normalization.
        if self.norm:
            x = tf.contrib.layers.instance_norm(x)
            
        if self.relu:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=0.2)
            
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        x = x * 0.1
        # try to test different scale
        return x + skip
    
    def _ResidualBlock_no_scale(self, x, filters):
        """Residual block a la ResNet"""
        skip = tf.identity(x)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        # try to use instance normalization.
        if self.norm:
            x = tf.contrib.layers.instance_norm(x)
        if self.relu:
            x = tf.nn.relu(x)
        else:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, kernel_initializer=self.init_kernel, padding='same')
        x = x * 0.2
        # try to test different scale
        return x + skip
        
    # three ways to do up-sampling
    def _upsample_by_depth_to_space(self, x, filters):
        """SubpixelConv"""
        x = tf.layers.conv2d(x, kernel_size=3, filters=filters, strides=1, padding='same')
        x = tf.depth_to_space(x, 2)
        #x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
        return x
    
    def _upsampling_layer(self, x):
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same',name='up_sample')
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
        return x
    
    def _upsampling_nearest(self, x, scale_factor=2):
        _, h, w, _ = x.get_shape().as_list()
        new_size = [h * scale_factor, w * scale_factor]
        return tf.image.resize_nearest_neighbor(x, size=new_size)
    
    def _hw_flatten(self, x) :
        if self.predict:
            return tf.reshape(x, shape=[self.batch_size, -1, x.shape[-1]])
            #return tf.reshape(x, shape=[4, -1, x.shape[-1]])
        else:
            return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])
           
    def _self_attention(self, x, channels, n):
        with tf.variable_scope("self_attention{}".format(n)) as scope:
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = tf.layers.conv2d(x, filters=channels//8, kernel_initializer=self.init_kernel, kernel_size=1, strides=1, padding='same')
            f = tf.layers.max_pooling2d(f, pool_size=2, strides=2, padding='SAME')
            g = tf.layers.conv2d(x, filters=channels//8, kernel_initializer=self.init_kernel, kernel_size=1, strides=1, padding='same')
            h = tf.layers.conv2d(x, filters=channels//2, kernel_initializer=self.init_kernel, kernel_size=1, strides=1, padding='same')
            h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='SAME')
            s = tf.matmul(self._hw_flatten(g), self._hw_flatten(f), transpose_b=True)
            beta = tf.nn.softmax(s)
            o = tf.matmul(beta, self._hw_flatten(h))
            gamma = tf.get_variable("gamma{}".format(n), [1], initializer=tf.constant_initializer(0.0))
            if self.predict:
                o = tf.reshape(o, shape=[self.batch_size, 96, 96, num_channels//2])
                #o = tf.reshape(o, shape=[4, 128, 128, num_channels//2])
            else:
                o = tf.reshape(o, shape=[batch_size, height, width, num_channels//2])
            o = tf.layers.conv2d(o, filters=channels, kernel_initializer=self.init_kernel, kernel_size=1, strides=1,padding='same')
            x = gamma * o + x
        return x
        
    def _encoder (self, x, f):
        x = tf.layers.conv2d(x, kernel_size=3, filters=f, strides=1, kernel_initializer=self.init_kernel, padding='same')
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x
        
    def _decoder(self, x, f):
        x = tf.layers.conv2d_transpose(x, filters=f, kernel_size=3, kernel_initializer=self.init_kernel,strides=2, padding='same')
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x
    
    '''
    def forward(self, x1, x2, x3=0):
        with tf.variable_scope('generator') as scope:
            x_branch1 = tf.layers.conv2d(x1, kernel_size=3, filters=64, kernel_initializer=self.init_kernel, strides=2, padding='same')
            #x_branch1 = tf.layers.max_pooling2d(x_branch1, pool_size=2, strides=2, padding='SAME')
            x_branch1 = tf.nn.relu(x_branch1)
            # 32*32*6
            x_branch2 = tf.layers.conv2d(x2, kernel_size=3, filters=64, kernel_initializer=self.init_kernel, strides=1, padding='same')
            x_branch2 = tf.nn.relu(x_branch2)
            x_start = tf.concat([x_branch1, x_branch2],axis=-1)
            x_start = tf.layers.conv2d(x_start, kernel_size=1, filters=64, kernel_initializer=self.init_kernel, strides=1, padding='same')
            # usually skip connected to conv without activation.
            # x_start = tf.nn.leaky_relu(x_start, alpha=.2)
            
            x = self._dense_residual_channel_attention(x_start, filters=64)            
            x = x_start + x
            x = self._upsample_by_depth_to_space(x, 256)
            if self.run_60:
                return x3
            else:
                #x = tf.layers.conv2d(x_super, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                #x = tf.nn.leaky_relu(x, alpha=.2)
                x = tf.layers.conv2d(x, filters=6, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                return x
                #return x + bright_2_end
    '''
    '''
    def forward(self,x1, x2, x3=0):
        #Builds the forward pass network graph
        with tf.variable_scope('generator') as scope:
            #if self.run_60:
            #    x_start = tf.concat([x1, x2, x3],axis=-1) 
            #else:
            
            # brightness reservation           
            channel_1 = x1.get_shape()[-1]
            bright_1 = tf.reshape(tf.reduce_mean(x1, axis=[1, 2]), (-1, 1, 1, channel_1))
            channel_2 = x2.get_shape()[-1]
            bright_2 = tf.reshape(tf.reduce_mean(x2, axis=[1, 2]), (-1, 1, 1, channel_2))
            if self.predict:
                base_1 = tf.fill([33, 64 , 64, 4], 1.0)
                base_2_start = tf.fill([33, 32 , 32, 6], 1.0)
                base_2_end = tf.fill([33, 64 , 64, 6], 1.0)
            else:
                base_1 = tf.fill([self.batch_size, 64 , 64, 4], 1.0)
                base_2_start = tf.fill([self.batch_size, 32 , 32, 6], 1.0)
                base_2_end = tf.fill([self.batch_size, 64 , 64, 6], 1.0)
            bright_1 = tf.multiply(base_1, bright_1)
            bright_2_start = tf.multiply(base_2_start, bright_2)
            bright_2_end = tf.multiply(base_2_end, bright_2)
            x1 = x1 - bright_1
            x2 = x2 - bright_2_start
            
            x_branch1 = tf.layers.conv2d(x1, kernel_size=3, filters=64, strides=1, padding='same')
            x_branch1 = tf.layers.max_pooling2d(x_branch1, pool_size=2, strides=2, padding='SAME')
            x_branch1 = tf.nn.leaky_relu(x_branch1, alpha=.2)
            # 32*32*6
            x_branch2 = tf.layers.conv2d(x2, kernel_size=3, filters=64, strides=1, padding='same')
            x_branch2 = tf.nn.leaky_relu(x_branch2, alpha=.2)
            x_start = tf.concat([x_branch1, x_branch2],axis=-1)
            x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=64, strides=1, padding='same')
            x_start = tf.nn.leaky_relu(x_start, alpha=.2)
            x = self.RRDB(x_start)
            x = self.RRDB(x)
            x = self.RRDB(x)
            x = self.RRDB(x)
            #x = self._self_attention(x, 128)           
            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, kernel_initializer=self.init_kernel, padding='same')           
            x = x + x_start
            # here the number should be 256
            # if 3 times should be 9 * 64
            x = self._upsample_by_depth_to_space(x, 128)
            #x = self._upsampling_nearest(x, 2)           
            if self.run_60:
                return x3
            else:
                #x = tf.layers.conv2d(x, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                #x = tf.nn.leaky_relu(x, alpha=.2)
                #x = tf.layers.conv2d(x_super, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                #x = tf.nn.leaky_relu(x, alpha=.2)
                x = tf.layers.conv2d(x, filters=6, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                return x
                #return x + bright_2_end
    '''
    '''           
    # apply u net to 10m bands.
    def forward(self,x1, x2, x3=0):
        #Builds the forward pass network graph
        with tf.variable_scope('generator') as scope:
            #if self.run_60:
            #    x_start = tf.concat([x1, x2, x3],axis=-1) 
            #else:
            
            # brightness reservation
            
            channel_1 = x1.get_shape()[-1]
            bright_1 = tf.reshape(tf.reduce_mean(x1, axis=[1, 2]), (-1, 1, 1, channel_1))
            channel_2 = x2.get_shape()[-1]
            bright_2 = tf.reshape(tf.reduce_mean(x2, axis=[1, 2]), (-1, 1, 1, channel_2))
            if self.predict:
                base_1 = tf.fill([33, 64 , 64, 4], 1.0)
                base_2_start = tf.fill([33, 32 , 32, 6], 1.0)
                base_2_end = tf.fill([33, 64 , 64, 6], 1.0)
            else:
                base_1 = tf.fill([self.batch_size, 64 , 64, 4], 1.0)
                base_2_start = tf.fill([self.batch_size, 32 , 32, 6], 1.0)
                base_2_end = tf.fill([self.batch_size, 64 , 64, 6], 1.0)
            bright_1 = tf.multiply(base_1, bright_1)
            bright_2_start = tf.multiply(base_2_start, bright_2)
            bright_2_end = tf.multiply(base_2_end, bright_2)
            x1 = x1 - bright_1
            x2 = x2 - bright_2_start
                    
            # decoder encoder to process 10m band
            # 64*64*4            
            x_branch1_1 = self._encoder(x1,64)
            # 32*32*64
            x_branch1_2 = self._encoder(x_branch1_1, 128)
            # 16*16*128
            x_branch1_3 = self._encoder(x_branch1_2, 256)
            # 8*8*256
            x_branch1_4 = self._encoder(x_branch1_3, 512)
            # 4*4*512
            x_branch1_5 = self._encoder(x_branch1_4, 512)
            # 2*2*512
            x_branch1_6 = self._decoder(x_branch1_5, 512)
            # 4*4*512
            x_branch1_7 = self._decoder(tf.concat([x_branch1_6, x_branch1_4],axis=-1), 256)
            # 8*8*256
            x_branch1_8 = self._decoder(tf.concat([x_branch1_7, x_branch1_3],axis=-1), 128)
            # 16*16*128
            x_branch1_9 = self._decoder(tf.concat([x_branch1_8, x_branch1_2],axis=-1), 64)
            # 32*32*64
            x_branch1_1ast = self._decoder(tf.concat([x_branch1_9, x_branch1_1],axis=-1), 32)
            # 64*64*32
            
            # 32*32*6
            x_branch2 = tf.layers.conv2d(x2, kernel_size=3, filters=64, strides=1, padding='same')            
            #x_start = tf.concat([x_branch1, x_branch2],axis=-1)
            x = self.RRDB(x_branch2)
            x = self.RRDB(x)
            x = self.RRDB(x)
            #x = self._self_attention(x, 128)           
            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, kernel_initializer=self.init_kernel, padding='same')           
            x = x + x_branch2
            
            x_branch2_last = self._upsample_by_depth_to_space(x, 128)
            #x = self._upsampling_nearest(x, 2)
            
            x_super = tf.concat([x_branch1_1ast, x_branch2_last], axis=-1)
            
            if self.run_60:
                return x3
            else:
                #x = tf.layers.conv2d(x, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                #x = tf.nn.leaky_relu(x, alpha=.2)
                x = tf.layers.conv2d(x_super, filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                x = tf.nn.leaky_relu(x, alpha=.2)
                x = tf.layers.conv2d(x, filters=6, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
                #return x
                return x + bright_2_end
    ''' 
    """
    def forward(self, x1=0, x2=0, x3=0):
        with tf.variable_scope('generator') as scope:           
            if self.run_60:
                x_start = tf.concat([x1, x2, x3],axis=-1)
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start = tf.nn.relu(x_start)
            else:
                x2 = tf.image.resize_bicubic(x2, [32, 32])
                x_start = tf.concat([x1, x2],axis=-1)
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')           
                x_start = tf.nn.relu(x_start)
            
            x = x_start
            #share_weight = tf.get_variable("share_weight", [1], initializer=tf.constant_initializer(0.0))
            for i in range(6):
                x = self._ResidualBlock(x, 128)
            #+ share_weight * x_start
            #x = x + x_start
            # try to add here a skip connection
            #x = self._self_attention(x, 128)
            #x_attention = self._self_attention(x, 128)            
            if self.run_60:
                x = tf.layers.conv2d(x, kernel_size=3, filters=2, kernel_initializer=self.init_kernel, strides=1, padding='same') 
                return x + x3
            else:
                #x = self._upsample_by_depth_to_space(x, 256)
                x = tf.layers.conv2d(x, kernel_size=3, filters=6, kernel_initializer=self.init_kernel, strides=1, padding='same')
                return x + x2
    """
    #dsen2 plus self attention!
    # separate input is useful
    # self attention is useful
    # max pool don not know.
    # channel attention to do.
    
    def forward(self, x1=0, x2=0, x3=0):
        with tf.variable_scope('generator') as scope:
            if self.graph_created:
                scope.reuse_variables()
            #brightness reservation   
            self.graph_created = True
                       
            channel_1 = x1.get_shape()[-1]
            bright_1 = tf.reshape(tf.reduce_mean(x1, axis=[1, 2]), (-1, 1, 1, channel_1))            
            channel_2 = x2.get_shape()[-1]
            bright_2 = tf.reshape(tf.reduce_mean(x2, axis=[1, 2]), (-1, 1, 1, channel_2))           
            if self.run_60:
                channel_3 = x3.get_shape()[-1]
                bright_3 = tf.reshape(tf.reduce_mean(x3, axis=[1, 2]), (-1, 1, 1, channel_3))
                                       
            if self.run_60:
                base_1 = tf.fill([self.batch_size, 96 , 96, 4], 1.0)
                base_2 = tf.fill([self.batch_size, 96 , 96, 6], 1.0)
                base_3 = tf.fill([self.batch_size, 96 , 96, 2], 1.0)
            else: 
                base_1 = tf.fill([self.batch_size, 64 , 64, 4], 1.0)
                base_2 = tf.fill([self.batch_size, 64 , 64, 6], 1.0)
            
            bright_1 = tf.multiply(base_1, bright_1)
            x1 = x1 - bright_1
            bright_2 = tf.multiply(base_2, bright_2)
            x2 = x2 - bright_2           
            if self.run_60:
                bright_3 = tf.multiply(base_3, bright_3)
                x3 = x3 - bright_3           
                        
            if self.run_60:
                x_start_1 = tf.layers.conv2d(x1, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel,padding='same')
                x_start_1 = tf.nn.leaky_relu(x_start_1, alpha=0.2)
                
                x_start_2 = tf.layers.conv2d(x2, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start_2 = tf.nn.leaky_relu(x_start_2, alpha=0.2)
                
                x_start_3 = tf.layers.conv2d(x3, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start_3 = tf.nn.leaky_relu(x_start_3, alpha=0.2)
                
                x_start = tf.concat([x_start_1, x_start_2, x_start_3],axis=-1)
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                #x_start = tf.nn.leaky_relu(x_start, alpha=0.2)
            else:
                x_start_1 = tf.layers.conv2d(x1, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel,padding='same')
                x_start_1 = tf.nn.leaky_relu(x_start_1, alpha=0.2)
                x_start_2 = tf.layers.conv2d(x2, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start_2 = tf.nn.leaky_relu(x_start_2, alpha=0.2)
                #x_start = tf.concat([x1, x2],axis=-1)
                x_start = tf.concat([x_start_1, x_start_2],axis=-1)
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')           
                #x_start = tf.nn.leaky_relu(x_start, alpha=0.2)
                
            x = x_start
            for i in range(3):
                x = self._ResidualBlock(x, 128)                           
            x = x + x_start
            # try to add here a skip connection
            #x = self._self_attention(x, 128)
            x_attention = self._self_attention(x, 128, 1)           
            x = x_attention
            for i in range(3):
                x = self._ResidualBlock(x, 128)   
            # try to add here a skip connection
            #x = x + x_start
            x = x + x_attention
            if self.run_60:
                x = tf.layers.conv2d(x, kernel_size=3, filters=2, kernel_initializer=self.init_kernel, strides=1, padding='same') 
                return x + x3 + bright_3
            else:
                x = tf.layers.conv2d(x, kernel_size=3, filters=6, kernel_initializer=self.init_kernel, strides=1, padding='same')
                return x + x2 + bright_2
    
    '''
    def forward(self, x1, x2):
        """Builds the forward pass network graph"""
        with tf.variable_scope('generator') as scope:
            x_start = tf.layers.conv2d(x1, kernel_size=3, filters=64, strides=1, padding='same')
            #x_start = tf.nn.leaky_relu(x_start, alpha=0.2, name='leakyReLU')
            x = self.RRDB(x_start)
            
            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
                   
            #x = self._upsampling_layer(x)
            #x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            #x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')  

            x_start_2 = tf.layers.conv2d(x2, kernel_size=3, filters=64, strides=1, padding='same')
            
            x = tf.concat([x, x_start_2],axis=-1)    
            # B x ResidualBlocks
            for i in range(4):
                x = self.ResidualBlock(x, kernel_size=3, filters=128, strides=1)     
            #x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            #x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU') 
            #x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            #x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU') 
            x = tf.layers.conv2d(x, kernel_size=3, filters=6, strides=1, padding='same', name='forward')
            return x + x2
    '''
    
    '''
    def forward(self, x1, x2):
        """Builds the forward pass network graph"""
        with tf.variable_scope('generator') as scope:
            x = tf.concat([x1, x2],axis=-1) 
            x_start = tf.layers.conv2d(x, kernel_size=3, filters=64, kernel_initializer=self.init_kernel, strides=1, padding='same')
            #x_start = tf.nn.leaky_relu(x_start, alpha=0.2, name='leakyReLU')
            x = self.RRDB(x_start)
            #x = self.RRDB(x)
            #x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            #x = x * 0.2 + x_start
            #hr_output = tf.layers.conv2d(x, kernel_size=3, filters=6,strides=1, padding='same')
               
            # B x ResidualBlocks
            #for i in range(4):
            #    x = self.ResidualBlock(x, kernel_size=3, filters=128, strides=1)     
            #x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            #x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU') 
            x = tf.layers.conv2d(x, kernel_size=3, filters=6, strides=1, kernel_initializer=self.init_kernel, padding='same', name='forward')
            x = x + x2
            return x
    '''
    '''
    def forward(self, x1, x2):
        """Builds the forward pass network graph"""
        with tf.variable_scope('generator') as scope:
            x_start_branch_2 = tf.layers.conv2d(x2, kernel_size=3, filters=64, kernel_initializer=self.init_kernel, strides=1, padding='same')
            #x_start = tf.nn.leaky_relu(x_start, alpha=0.2, name='leakyReLU')
            x_branch_2 = self.RRDB(x_branch_2)
            x_branch_2 = self.RRDB(x_branch_2)
            x_branch_2 = self.RRDB(x_branch_2)
            x_branch_2 = tf.layers.conv2d(x_branch_2, kernel_size=3, filters=64, strides=1, kernel_initializer=self.init_kernel, padding='same', name='forward')
            x_branch_2 = x_branch_2 + x_start_branch_2
            x_branch_2 = self._upsampling_layer(x_branch_2)
            
            #should we add conv here??
            #x = tf.layers.conv2d(x,filters=64, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
            #x = tf.nn.leaky_relu(x,alpha=.2)
            
            x_start_branch_1 = tf.layers.conv2d(x1, kernel_size=3, filters=64, kernel_initializer=self.init_kernel, strides=1, padding='same')
            #x_start = tf.nn.leaky_relu(x_start, alpha=0.2, name='leakyReLU')
            x_branch_1 = self.RRDB(x_branch_1)
            x_branch_1 = self.RRDB(x_branch_1)
            x_branch_1 = self.RRDB(x_branch_1)
            x_branch_1 = tf.layers.conv2d(x_branch_1, kernel_size=3, filters=64, strides=1, kernel_initializer=self.init_kernel, padding='same', name='forward')
            x_branch_1 = x_branch_1 + x_start_branch_1
                       
            x = tf.concat([x_branch_1, x_branch_2],axis=-1)
            
            x = tf.layers.conv2d(x,filters=128, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
            x = tf.nn.leaky_relu(x,alpha=.2)
            x = tf.layers.conv2d(x,filters=3, kernel_size=3, kernel_initializer=self.init_kernel, strides=1, padding='same')
            return x
    '''
    '''
    # sub pixel module, put it in the final appendix to show why it is not good.
    def forward(self, x1=0, x2=0, x3=0):
        with tf.variable_scope('generator') as scope:
            # brightness reservation
            channel_1 = x1.get_shape()[-1]
            bright_1 = tf.reshape(tf.reduce_mean(x1, axis=[1, 2]), (-1, 1, 1, channel_1))
            
            channel_2 = x2.get_shape()[-1]
            bright_2 = tf.reshape(tf.reduce_mean(x2, axis=[1, 2]), (-1, 1, 1, channel_2))
            
            if self.run_60:
                channel_3 = x3.get_shape()[-1]
                bright_3 = tf.reshape(tf.reduce_mean(x3, axis=[1, 2]), (-1, 1, 1, channel_3))
            
            if self.run_60:
                base_1 = tf.fill([self.batch_size, 96 , 96, 4], 1.0)
                base_2 = tf.fill([self.batch_size, 48 , 48, 6], 1.0)
                base_3 = tf.fill([self.batch_size, 16 , 16, 2], 1.0)
                base_3_out = tf.fill([self.batch_size, 96 , 96, 2], 1.0)
            else:
                base_1 = tf.fill([self.batch_size, 32 , 32, 4], 1.0)
                base_2 = tf.fill([self.batch_size, 16 , 16, 6], 1.0)
                base_2_out = tf.fill([self.batch_size, 32 , 32, 6], 1.0)
            
            bright_1 = tf.multiply(base_1, bright_1)
            x1 = x1 - bright_1
            bright_2_in= tf.multiply(base_2, bright_2)
            x2 = x2 - bright_2_in
            bright_2_out = tf.multiply(base_2_out, bright_2)
            
            if self.run_60:
                bright_3_in = tf.multiply(base_3, bright_3)
                x3 = x3 - bright_3_in
                bright_3_out = tf.multiply(base_3_out, bright_3)
                       
            if self.run_60:
                x_start_1 = tf.layers.conv2d(x1, kernel_size=3, filters=128, strides=6, kernel_initializer=self.init_kernel,padding='same')
                x_start_1 = tf.nn.leaky_relu(x_start_1, alpha=0.2, name='leakyReLU')
                x_start_2 = tf.layers.conv2d(x2, kernel_size=3, filters=128, strides=3, kernel_initializer=self.init_kernel, padding='same')
                x_start_2 = tf.nn.leaky_relu(x_start_2, alpha=0.2, name='leakyReLU')
                x_start_3 = tf.layers.conv2d(x3, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start_3 = tf.nn.leaky_relu(x_start_3, alpha=0.2, name='leakyReLU')
                x_start = tf.concat([x_start_1, x_start_2, x_start_3],axis=-1)
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
            else:
                x_start_1 = tf.layers.conv2d(x1, kernel_size=3, filters=128, strides=2, kernel_initializer=self.init_kernel,padding='same')
                #x_start_1 = tf.nn.leaky_relu(x_start_1, alpha=0.2, name='leakyReLU')
                x_start_1 = tf.nn.relu(x_start_1)
                x_start_2 = tf.layers.conv2d(x2, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')
                x_start_2 = tf.nn.relu(x_start_2)
                x_start = tf.concat([x_start_1, x_start_2],axis=-1) 
                x_start = tf.layers.conv2d(x_start, kernel_size=3, filters=128, strides=1, kernel_initializer=self.init_kernel, padding='same')           
            
            share_weight = tf.get_variable("share_weight", [1], initializer=tf.constant_initializer(0.0))
            
            xx = self._self_attention(x_start, 128, 1)
            
            residual = xx
            
            for i in range(6):
                xx = residual*share_weight + self._ResidualBlock_no_scale(xx, 128)
                
            res = self._self_attention(xx, 128, 2)
            
            res = res + x_start
            
            x = self._upsample_by_depth_to_space(res, 512)
            
            if self.run_60: 
                x = tf.layers.conv2d(x, kernel_size=3, filters=2, kernel_initializer=self.init_kernel, strides=1, padding='same') 
                return x + bright_3_out
            else:
                #x = self._upsample_by_depth_to_space(x, 256)
                x = tf.layers.conv2d(x, kernel_size=3, filters=6, kernel_initializer=self.init_kernel, strides=1, padding='same') 
                #return x
                return x + bright_2_out
    '''
    def _content_loss(self, y, y_pred):
        #"""MSE, VGG22, or VGG54"""
        if self.content_loss == 'MSE':
            return tf.reduce_mean(tf.square(y - y_pred))
        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(y - y_pred))

    def _adversarial_loss(self, y, y_pred):
        y_discrim_logits_fake = self.discriminator.forward(y_pred)
        y_discrim_logits_real = self.discriminator.forward(y)
        if self.adversarial_loss == 'Vanila-gan':
            return self.adv_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_discrim_logits_fake, labels=tf.ones_like(y_discrim_logits_fake)))
        if self.adversarial_loss == 'wgan-gp':
            return -1 * self.adv_weight * tf.reduce_mean(y_discrim_logits_fake)
        if self.adversarial_loss == 'relativistic':
            loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_discrim_logits_real - tf.reduce_mean(y_discrim_logits_fake),labels=tf.zeros_like(y_discrim_logits_real)))
            loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_discrim_logits_fake - tf.reduce_mean(y_discrim_logits_real),labels=tf.ones_like(y_discrim_logits_fake)))
            return self.adv_weight * (loss_real + loss_real)
        if self.adversarial_loss == 'hinge':
            return self.adv_weight * -1 * tf.reduce_mean(y_discrim_logits_fake)
            
    def _perceptron_loss(self, y, y_pred):
        fake, _ = self.discriminator.forward(y_pred, perceptron=True)
        real, _ = self.discriminator.forward(y, perceptron=True )
        return 1e-1*tf.reduce_mean(tf.square(fake-real))
               
    def pretrain_loss(self, y, y_pred):
        #return tf.reduce_mean(tf.square(y - y_pred))
        return tf.reduce_mean(tf.abs(y - y_pred))       
        #temp = tf.square(y - y_pred)
        #return tf.reduce_mean(tf.sqrt(temp+1e-8))
        
    def valid_loss(self, y, y_pred): 
        return tf.reduce_mean(tf.square(y - y_pred))

class Discriminator:
    def __init__(self, loss_type = 'relativistic', image_size = 32, batch_size = 128, norm = False, run_60 = False):
        self.graph_created = False
        self.loss_type = loss_type
        #self.learning_rate = learning_rate
        self.image_size = image_size
        self.batch_size = batch_size
        self.init_kernel = tf.initializers.he_normal(seed=111)
        self.norm = norm
        self.run_60=run_60
        
    def ConvolutionBlock(self, x, kernel_size, filters, strides):
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides,kernel_initializer=self.init_kernel, padding='same', use_bias=False)
        if self.norm:
            x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
        return x

    def forward(self, x, perceptron = False):
        with tf.variable_scope('discriminator') as scope:
        # Reuse variables when graph is applied again
            if self.graph_created:
                scope.reuse_variables()

            self.graph_created = True

            # Image dimensions are fixed to the training size because of the FC layer
            if self.run_60:
                x.set_shape([None, self.image_size, self.image_size, 2])
            else:
                x.set_shape([None, self.image_size, self.image_size, 6])

            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, kernel_initializer=self.init_kernel, padding='same')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

            x = self.ConvolutionBlock(x, 3, 64, 2)
            x = self.ConvolutionBlock(x, 3, 128, 1)
            x = self.ConvolutionBlock(x, 3, 128, 2)
            x = self.ConvolutionBlock(x, 3, 128, 1)
            x = self.ConvolutionBlock(x, 3, 128, 2)
            #x = self.ConvolutionBlock(x, 3, 512, 1)
            #x = self.ConvolutionBlock(x, 3, 512, 2)

            x_flatten = tf.layers.flatten(x)
            x = tf.layers.dense(x_flatten, 256)
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
            logits = tf.layers.dense(x, 1)
            #x = tf.sigmoid(logits)
            if perceptron:
                return x_flatten, logits
            else:
                return logits
    
    def gradient_penalty(self, g_y_pred, label):        
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = label + alpha * (g_y_pred - label)
        gradients = tf.gradients(self.forward(interpolates), [interpolates])[0]
        grad_norm = tf.norm(tf.layers.flatten(gradients), axis=1)
        gp = 10 * tf.reduce_mean(tf.square(grad_norm - 1.))
        return gp
        
        
    def loss(self, y_real_pred_logits, y_fake_pred_logits):
        """Discriminator wants to maximize log(y_real) + log(1-y_fake)."""
        if self.loss_type == 'Vanila-gan':
            loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
            loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
            return loss_real + loss_fake
        if self.loss_type == 'wgan-gp':
            d_loss_ = tf.reduce_mean(y_fake_pred_logits) - tf.reduce_mean(y_real_pred_logits)
            return d_loss_ 
        if self.loss_type == 'relativistic':
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_real_pred_logits - tf.reduce_mean(y_fake_pred_logits),labels=tf.ones_like(y_real_pred_logits))) / 2
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_fake_pred_logits - tf.reduce_mean(y_real_pred_logits),labels=tf.zeros_like(y_fake_pred_logits))) / 2
            return d_loss_real + d_loss_fake
        if self.loss_type == 'hinge':
            loss_real = tf.reduce_mean(tf.nn.relu(1.0-y_real_pred_logits))
            loss_fake = tf.reduce_mean(tf.nn.relu(1.0+y_fake_pred_logits))
            return loss_real + loss_fake
            
    #try to use two time scale strategy 
    #def optimize(self, loss):
    #    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    #    #with tf.control_dependencies(update_ops):
    #    return hvd.DistributedOptimizer(tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0., beta2=0.9), compression= hvd.Compression.fp16).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
