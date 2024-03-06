import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop

class TensAIR(Model):
    def __init__(self):
        super(TensAIR, self).__init__()
        
    @tf.function
    def single_train_step(self, data):
        # Reset metrics (they are averaged in TensAIR)
        self.reset_metrics()
        
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        
        #calculate accuracy
        self.compiled_metrics.update_state(y, y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
                
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        #Update delta gradients
        for i, d in enumerate(self.delta):
            if gradients[i] is not None:
                d.assign_add(gradients[i])
        
        #Update delta metrics
        self.delta_loss.assign_add(loss)
        self.delta_accuracy.assign_add(self.metrics[1].result())

        return self.metrics[0].result(),self.metrics[1].result()
    
    def init_delta(self, trainable_weights):
        #Delta from last broadcast
        self.delta = [tf.Variable(tf.zeros(tw.shape, dtype=float), trainable=False) for tw in trainable_weights]
        self.delta_accuracy = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.delta_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        return
    
    @tf.function
    def retrieve_delta(self, empty):
        return self.delta_loss,self.delta_accuracy,*self.delta
    
    @tf.function
    def prediction(self, data):
        self.reset_metrics()
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        return loss, y_pred
    
    @tf.function
    def apply_gradient(self, *g):
        gradients = list(g)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                 
        return {'error' : 0}
    
    @tf.function
    def clear_delta(self, empty):
        
        for i, d in enumerate(self.delta):
            d.assign(tf.zeros(d.shape, dtype=float))
        
        self.delta_accuracy.assign(0)
        self.delta_loss.assign(0)
        return 0
    
    
    @tf.function
    def save_variables(self, file):
        #file_str = compat.as_text(file.numpy())
        #tf.train.Checkpoint(step=self.variables).write(tf.eval(file))
        tf.train.Checkpoint(step=self.variables).write("trained_variables/variables")
        return 0
    
    
    @tf.function
    def evaluation(self, data):
        
        self.reset_metrics()
        self.reset_states()
        
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        
        y_pred = self(x, training=False)  
        loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        
        self.compiled_metrics.update_state(y, y_pred)
        
        
        return {m.name: m.result() for m in self.metrics}
    
# Custom key function that returns the tensor's name part relevant for sorting
def identity_string_key(tensor):
    # Simply return the part of the name that includes "Identity"
    return tensor.name.split(':')[0]

def fill_tuple(structure, lst):
    if isinstance(structure, tuple):
        filled_structure = []
        for item in structure:
            filled_structure.append(fill_tuple(item, lst))
        return tuple(filled_structure)
    else:
        return lst.pop(0)

def define_signatures(model, input_tensors_dims, input_tensors_types, input_tensors_structure):
    save_variables = model.save_variables.get_concrete_function(tf.TensorSpec([], tf.int32, name='file'))
    clear_delta = model.clear_delta.get_concrete_function(tf.TensorSpec([], tf.int32, name='empty'))
    retrieve_delta = model.retrieve_delta.get_concrete_function(tf.TensorSpec([], tf.int32, name='empty2'))
    
    #automatically find shape of input tensors for apply_gradient
    retrieve_delta_output = retrieve_delta.outputs #the deltas retrieved are the gradients that will be applied
    indexed_items = [(index, item) for index, item in enumerate(retrieve_delta_output[2:])]#create index to store order of the list (and remove loss and acc metrics)
    delta_output_sorted = sorted(indexed_items, key=lambda x: identity_string_key(x[1])) #saved_model_cli sort by name alphabetically
    delta_output_well_named = [(i[0], tf.TensorSpec(i[1].shape, i[1].dtype, name=f"apply_gradient_{index:0{5}d}")) for index, i in enumerate(delta_output_sorted)] #create tensor_shapes in the correct order
    delta_output_original_order = [item for index, item in sorted(delta_output_well_named, key=lambda x: x[0])] #return list to original order
    apply_gradient = model.apply_gradient.get_concrete_function(*delta_output_original_order)
    
    evaluate_specs = []
    prediction_specs = []
    train_specs = []

    for index, (dim, t) in enumerate(zip(input_tensors_dims,input_tensors_types)):
        evaluate_specs.append(tf.TensorSpec(dim, t, name=f"evaluate_{index:0{5}d}"))
        prediction_specs.append(tf.TensorSpec(dim, t, name=f"predict_{index:0{5}d}"))
        train_specs.append(tf.TensorSpec(dim, t, name=f"train_{index:0{5}d}"))

    evaluate = model.evaluation.get_concrete_function(fill_tuple(input_tensors_structure,evaluate_specs))
    prediction = model.prediction.get_concrete_function(fill_tuple(input_tensors_structure,prediction_specs))
    train_step = model.single_train_step.get_concrete_function(fill_tuple(input_tensors_structure,train_specs))
    
    signatures={'apply_gradient': apply_gradient, 'evaluate': evaluate, 'save': save_variables, 'prediction': prediction, 'clear_delta': clear_delta, 'train_step':train_step, 'retrieve_delta':retrieve_delta}
    
    return signatures
    
    