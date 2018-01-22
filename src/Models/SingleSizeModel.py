"""
"""
import tensorflow as tf
from tensorflow.python.layers.convolutional import conv2d
from src.Utils import get_img_batch
import glob
class SingleSizeModel():
    """ A convolution model that handles only same size cover
    and secret images.
    """
    def get_prep_network_op(self,secret_tensor):

        with tf.variable_scope('prep_net'):

            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = conv2d(inputs=secret_tensor,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = conv2d(inputs=secret_tensor,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)           
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = conv2d(inputs=secret_tensor,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)           
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)

            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')

            conv_5x5 = conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)

            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')

            return concat_final


    def get_hiding_network_op(self,cover_tensor,prep_output):

        with tf.variable_scope('hide_net'):
            concat_input = tf.concat([cover_tensor,prep_output],axis=3,name='images_features_concat')

            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = conv2d(inputs=concat_input,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = conv2d(inputs=concat_input,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)          
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = conv2d(inputs=concat_input,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)          
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)

            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')

            conv_5x5 = conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)

            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')
            output = tf.layers.conv2d(inputs=concat_final,filters=3,kernel_size=1,padding='same',name='output')

            return output



    def get_reveal_network_op(self,container_tensor):

        with tf.variable_scope('reveal_net'):

            with tf.variable_scope("3x3_conv_branch"):
                conv_3x3 = conv2d(inputs=container_tensor,filters=50,kernel_size=3,padding='same',name="1",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="2",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="3",activation=tf.nn.relu)
                conv_3x3 = conv2d(inputs=conv_3x3,filters=50,kernel_size=3,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("4x4_conv_branch"):
                conv_4x4 = conv2d(inputs=container_tensor,filters=50,kernel_size=4,padding='same',name="1",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="2",activation=tf.nn.relu)          
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="3",activation=tf.nn.relu)
                conv_4x4 = conv2d(inputs=conv_4x4,filters=50,kernel_size=4,padding='same',name="4",activation=tf.nn.relu)

            with tf.variable_scope("5x5_conv_branch"):
                conv_5x5 = conv2d(inputs=container_tensor,filters=50,kernel_size=5,padding='same',name="1",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="2",activation=tf.nn.relu)           
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="3",activation=tf.nn.relu)
                conv_5x5 = conv2d(inputs=conv_5x5,filters=50,kernel_size=5,padding='same',name="4",activation=tf.nn.relu)

            concat_1 = tf.concat([conv_3x3,conv_4x4,conv_5x5],axis=3,name='concat_1')

            conv_5x5 = conv2d(inputs=concat_1,filters=50,kernel_size=5,padding='same',name="final_5x5",activation=tf.nn.relu)
            conv_4x4 = conv2d(inputs=concat_1,filters=50,kernel_size=4,padding='same',name="final_4x4",activation=tf.nn.relu)
            conv_3x3 = conv2d(inputs=concat_1,filters=50,kernel_size=3,padding='same',name="final_3x3",activation=tf.nn.relu)

            concat_final = tf.concat([conv_5x5,conv_4x4,conv_3x3],axis=3,name='concat_final')

        output = tf.layers.conv2d(inputs=concat_final,filters=3,kernel_size=1,padding='same',name='output')

        return output

    def get_noise_layer_op(self,tensor,std=.1):
        with tf.variable_scope("noise_layer"):
            return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32) 

    def get_loss_op(self,secret_true,secret_pred,cover_true,cover_pred,beta=.5):

        with tf.variable_scope("losses"):
            beta = tf.constant(beta,name="beta")
            secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
            cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)
            final_loss = cover_mse + beta*secret_mse
            return final_loss , secret_mse , cover_mse 

    def get_tensor_to_img_op(self,tensor):
        with tf.variable_scope("",reuse=True):
            t = tensor*tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
            return tf.clip_by_value(t,0,1)
    
    def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
    
        prep_output_op = self.get_prep_network_op(secret_tensor)
        hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)
        noise_add_op = self.get_noise_layer_op(hiding_output_op)
        reveal_output_op = self.get_reveal_network_op(noise_add_op)

        loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hiding_output_op,beta=self.beta)

        minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op,global_step=global_step_tensor)

        tf.summary.scalar('loss', loss_op,family='train')
        tf.summary.scalar('reveal_net_loss', secret_loss_op,family='train')
        tf.summary.scalar('cover_net_loss', cover_loss_op,family='train')

        tf.summary.image('secret',self.get_tensor_to_img_op(secret_tensor),max_outputs=1,family='train')
        tf.summary.image('cover',self.get_tensor_to_img_op(cover_tensor),max_outputs=1,family='train')
        tf.summary.image('hidden',self.get_tensor_to_img_op(hiding_output_op),max_outputs=1,family='train')
        tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
        tf.summary.image('revealed',self.get_tensor_to_img_op(reveal_output_op),max_outputs=1,family='train')

        merged_summary_op = tf.summary.merge_all()

        return minimize_op, merged_summary_op 
    
    def prepare_test_graph(self,secret_tensor,cover_tensor):
        with tf.variable_scope("",reuse=True):

            prep_output_op = self.get_prep_network_op(secret_tensor)
            hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)
            reveal_output_op = self.get_reveal_network_op(hiding_output_op)

            loss_op,secret_loss_op,cover_loss_op = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hiding_output_op)

            tf.summary.scalar('loss', loss_op,family='test')
            tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
            tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')

            tf.summary.image('secret',self.get_tensor_to_img_op(secret_tensor),max_outputs=1,family='test')
            tf.summary.image('cover',self.get_tensor_to_img_op(cover_tensor),max_outputs=1,family='test')
            tf.summary.image('hidden',self.get_tensor_to_img_op(hiding_output_op),max_outputs=1,family='test')
            tf.summary.image('revealed',self.get_tensor_to_img_op(reveal_output_op),max_outputs=1,family='test')

            merged_summary_op = tf.summary.merge_all()

            return merged_summary_op 
    
    def prepare_deployment_graph(self,secret_tensor,cover_tensor,covered_tensor):
        with tf.variable_scope("",reuse=True):

            prep_output_op = self.get_prep_network_op(secret_tensor)
            hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,prep_output=prep_output_op)

            reveal_output_op = self.get_reveal_network_op(covered_tensor)

            return hiding_output_op ,  reveal_output_op
        
    def get_tensor_to_img_op(self,tensor):
        with tf.variable_scope("",reuse=True):
            t = tensor*tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
            return tf.clip_by_value(t,0,1)
    
    
    def __init__(self, beta,log_path, input_shape=(None,224, 224, 3) ):
        
        self.beta = beta
        self.learning_rate = 0.0001
        self.sess = tf.InteractiveSession()
        
        self.secret_tensor = tf.placeholder(shape=input_shape,dtype=tf.float32,name="input_prep")
        self.cover_tensor = tf.placeholder(shape=input_shape,dtype=tf.float32,name="input_hide")
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

        self.train_op , self.summary_op = self.prepare_training_graph(self.secret_tensor,self.cover_tensor,self.global_step_tensor)

        self.writer = tf.summary.FileWriter(log_path,self.sess.graph)

        self.test_op = self.prepare_test_graph(self.secret_tensor,self.cover_tensor)

        self.covered_tensor = tf.placeholder(shape=input_shape,dtype=tf.float32,name="deploy_covered")
        self.deploy_hide_image_op , self.deploy_reveal_image_op = self.prepare_deployment_graph(self.secret_tensor,self.cover_tensor,self.covered_tensor)
        self.sess.run(tf.global_variables_initializer())
        
        print("OK")
    
    def make_chkp(self,path):
        saver = tf.train.Saver(max_to_keep=1)
        global_step = self.sess.run(self.global_step_tensor)
        saver.save(self.sess,path,global_step)

    def load_chkp(self,path):
        saver = tf.train.Saver(max_to_keep=1)
        global_step = self.sess.run(self.global_step_tensor)
        print("LOADED")
        saver.restore(self.sess,path)
        
        
    def train(self,steps,files_list,batch_size):
        

        for step in range(steps):
            covers,secrets = get_img_batch(files_list=files_list,batch_size=batch_size)
            self.sess.run([self.train_op],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
            
            if step %10 == 0:
                summary,global_step = self.sess.run([self.summary_op,self.global_step_tensor],feed_dict={"input_prep:0":secrets,"input_hide:0":covers})
                self.writer.add_summary(summary,global_step)


                
            
        
m = SingleSizeModel(beta=.25,log_path="/valohai/outputs/")
# m.load_chkp("/home/harsh/ml/Stegano/checkpoints/beta_0.75.chkp-102192")
files_list = glob.glob("/valohai/inputs/training-set-images/train/"+"**/*")
# print(files_list)
for i in range(100):
    m.make_chkp('/valohai/outputs/beta_.25.chkp')
    m.train(1000,files_list,8)
    print("Saved")

# print(files_list)

