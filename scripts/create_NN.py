import tensorflow as tf 

def create_NN_model(input_shape,output_shape,n_layers,n_nodes,activation,L1_penalty,L2_penalty,use_batch_norm):
    X_input = tf.keras.layers.Input(shape=input_shape)
    X = X_input
    for i in range(n_layers):
        X = tf.keras.layers.Dense(n_nodes,activation=activation,kernel_regularizer=tf.keras.regularizers.L1L2(L1_penalty,L2_penalty))(X)
        if use_batch_norm==True:
            X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    output = tf.keras.layers.Dense(output_shape,activation='linear',kernel_regularizer=tf.keras.regularizers.L1L2(L1_penalty,L2_penalty))(X)
    
    model = tf.keras.Model(inputs=X_input,outputs=output)
    return model