import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.layers.experimental import preprocessing as PP
import tensorflow as tf

import kerastuner as kt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
raw_data = pd.read_csv('train.csv')

orig_vector_data = raw_data[
    ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12',
     'cont13', 'cont14']]  #
orig_label_data = raw_data[['target']]

#z_scores = stats.zscore(raw_data)
#abs_z_scores = np.abs(z_scores)
#filtered_entries = (abs_z_scores < 1).all(axis=1)
#raw_data_scrubbed = raw_data[filtered_entries]

#raw_data_scrubbed = raw_data[(np.abs(stats.zscore(raw_data['target'])) < 2)]

outlier_band = (np.quantile(raw_data.target,0.75) - np.quantile(raw_data.target,0.25))*1.5
low, high = np.quantile(raw_data.target,0.25) - outlier_band, np.quantile(raw_data.target,0.75) + outlier_band
raw_data_scrubbed = raw_data[ (raw_data.target>low) & (raw_data.target<high)]


outlier_band = (np.quantile(raw_data_scrubbed.cont7,0.75) - np.quantile(raw_data_scrubbed.cont7,0.25))*1.5
low, high = np.quantile(raw_data_scrubbed.cont7, 0.25) - outlier_band, np.quantile(raw_data_scrubbed.cont7,0.75) + outlier_band
raw_data_scrubbed = raw_data_scrubbed[(raw_data_scrubbed.cont7>low) & (raw_data_scrubbed.cont7<high)]

outlier_band = (np.quantile(raw_data_scrubbed.cont9,0.75) - np.quantile(raw_data_scrubbed.cont9,0.25))*1.5
low, high = np.quantile(raw_data_scrubbed.cont9, 0.25) - outlier_band, np.quantile(raw_data_scrubbed.cont9,0.75) + outlier_band
raw_data_scrubbed= raw_data_scrubbed[(raw_data_scrubbed.cont9>low) & (raw_data_scrubbed.cont9<high)]

vector_data = raw_data_scrubbed[
    ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12',
     'cont13', 'cont14']]  #
label_data = raw_data_scrubbed[['target']]

training_vector_data = vector_data
training_label_data = label_data


def normalizedataframe(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


normalizer = PP.Normalization()
normalizer.adapt(training_vector_data.to_numpy())


# training_vector_data = normalizedataframe(training_vector_data)
# training_label_data = normalizedataframe(training_label_data)
# validation_vector_data = normalizedataframe(validation_vector_data)
# validation_label_data = normalizedataframe(validation_label_data)

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.squared_difference(y_true, y_pred)))


def model_builder(hp):
    global normalizer

    this_model = tf.keras.Sequential()
    this_model.add(normalizer)

    hp_layer_one_units = hp.Int('layer_one_units', min_value=2, max_value=16, step=1)
    this_model.add(tf.keras.layers.Dense(units=hp_layer_one_units, activation='relu'))
    hp_layer_two_units = hp.Int('layer_two_units', min_value=2, max_value=16, step=1)
    this_model.add(tf.keras.layers.Dense(units=hp_layer_two_units, activation='relu'))

    this_model.add(tf.keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
    hp_momentum = hp.Int('momentum', min_value=1, max_value=9, step=1)

    tf.keras.optimizers.SGD(learning_rate=hp_learning_rate, momentum=hp_momentum / 10)
    this_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                       loss=root_mean_squared_error,
                       metrics=['mean_squared_error', 'accuracy'])

    return this_model


tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=10,
                     factor=2,
                     directory='my_dir',
                     project_name='intro_to_kt9')

tuner.search(training_vector_data.values, training_label_data.values, epochs=50,
             validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=100)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('layer_one_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
model.fit(training_vector_data, training_label_data, epochs=10,
          validation_split=0.2)

# test_loss, test_acc = model.evaluate(validation_vector_data.values, validation_label_data.values, verbose=2)

# predictions  = model.predict(validation_vector_data.values)

model.save('Model2.h5')

raw_data = pd.read_csv('test.csv')
raw_data.astype({'id': 'int32'}).dtypes
vector_data = raw_data[
    ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12',
     'cont13', 'cont14']]  #

predictions = model.predict(vector_data.values)

id = []
target = []
result = []
x = 0
for index, row in raw_data.iterrows():
    result.append([int(row['id']), predictions[x][0]])
    x += 1

final_df = pd.DataFrame(result, columns=['id', 'target'])
final_df.to_csv('sult.csv', index=False)

final_df.head()
test_dataset = tf.data.Dataset.from_tensor_slices((orig_vector_data, orig_label_data))
test_dataset = tf.data.Dataset.from_tensor_slices((training_vector_data, training_label_data))

#result = model.evaluate(test_dataset)

print('lol')
