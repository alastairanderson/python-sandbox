#region packages
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#endregion

#region constants
NUM_OF_HOUSES = 160

MIN_HOUSE_SIZE = 1000  # sq ft
MAX_HOUSE_SIZE = 3500  # sq ft

MIN_HOUSE_PRICE = 20000
MAX_HOUSE_PRICE = 70000

RANDOM_SEED = 42

TRAINING_SET_SIZE = 0.7  # 70%
NUM_TRAINING_SAMPLES = math.floor(NUM_OF_HOUSES * TRAINING_SET_SIZE)

LEARNING_RATE = 0.1   # The gradient descent step size

NUM_TRAINING_ITERATIONS = 50
DISPLAY_EVERY_X_ITERATION = 2
#endregion

#region public methods
def run():

    # 1. Prepare the data
    house_size, house_price = __generate_data()
    # __visualise_data(house_size, house_price)
    train_size, train_size_norm, train_price, train_price_norm = __get_training_data(house_size, house_price)
    test_size, test_size_norm, test_price, test_price_norm = __get_test_data(house_size, house_price)

    # 2. Define the operation for predicting the values
    tf_house_size, tf_price = __configure_placeholders()
    tf_size_factor, tf_price_offset = __configure_model_variables()

    price_pred = __configure_model(tf_size_factor, tf_house_size, tf_price_offset)

    # 3. Define a cost function to measure the error of a particular model
    tf_cost_fn = __configure_loss_function(price_pred, tf_price)

    # 4. Define an optimiser to minimise the cost function
    optimizer = __train_model(tf_cost_fn)

    # Execute the graph
    init = tf.global_variables_initializer()

    # Launch the graph in the session
    with tf.Session() as sess:
        sess.run(init)

#region visuals
        # calculate the number of lines to animation
        fit_num_plots = math.floor(NUM_TRAINING_ITERATIONS/DISPLAY_EVERY_X_ITERATION)
        # add storage of factor and offset values from each epoch
        fit_size_factor = np.zeros(fit_num_plots)
        fit_price_offsets = np.zeros(fit_num_plots)
        fit_plot_idx = 0    
#endregion

        # keep iterating the training data
        for iteration in range(NUM_TRAINING_ITERATIONS):

            # Fit all training data
            for (x, y) in zip(train_size_norm, train_price_norm):
                sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

#region status + visuals
            # Display current status
            if (iteration + 1) % DISPLAY_EVERY_X_ITERATION == 0:
                c = sess.run(tf_cost_fn, feed_dict={tf_house_size: train_size_norm, tf_price:train_price_norm})
                print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                    "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

                # Save the fit size_factor and price_offset to allow animation of learning process
                fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
                fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
                fit_plot_idx = fit_plot_idx + 1
#endregion

        print("Optimization Finished!")
        training_cost = sess.run(tf_cost_fn, feed_dict={tf_house_size: train_size_norm, tf_price: train_price_norm})
        print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

#region visuals
        # Plot of training and test data, and learned regression
        
        # get values used to normalized data so we can denormalize data back to its original scale
        train_house_size_mean = train_size.mean()
        train_house_size_std = train_size.std()

        train_price_mean = train_price.mean()
        train_price_std = train_price.std()

        # Plot the graph
        def show_model_on_graph():
            plt.rcParams["figure.figsize"] = (10,8)
            plt.figure()
            plt.ylabel("Price")
            plt.xlabel("Size (sq.ft)")
            plt.plot(train_size, train_price, 'go', label='Training data')
            plt.plot(test_size, test_price, 'mo', label='Testing data')
            plt.plot(train_size_norm * train_house_size_std + train_house_size_mean,
                    (sess.run(tf_size_factor) * train_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
                    label='Learned Regression')
        
            plt.legend(loc='upper left')
            plt.show()

        # show_model_on_graph()

        def animate_gradient_descent_on_graph():
            # Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to 
            # find the values that returned the "best" fit line.
            fig, ax = plt.subplots()
            line, = ax.plot(house_size, house_price)
            # BUG: All points are being plotted with lines connecting them

            plt.rcParams["figure.figsize"] = (10,8)
            plt.title("Gradient Descent Fitting Regression Line")
            plt.ylabel("Price")
            plt.xlabel("Size (sq.ft)")
            plt.plot(train_size, train_price, 'go', label='Training data')
            plt.plot(test_size, test_price, 'mo', label='Testing data')

            def animate(i):
                # line.linestyle = '-'
                line.set_xdata(train_size_norm * train_house_size_std + train_house_size_mean)  # update the data
                line.set_ydata((fit_size_factor[i] * train_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
                return line,
        
            # Init only required for blitting to give a clean slate.
            def initAnim():
                line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
                return line,

            ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                        interval=1000, blit=True)

            plt.show()  

        animate_gradient_descent_on_graph()
         
#endregion

#endregion

#region utility methods
def __normalize(array):
    return (array - array.mean()) / array.std()
#endregion

#region data generation methods
def __generate_data():
    np.random.seed(RANDOM_SEED)
    house_size = np.random.randint(low=MIN_HOUSE_SIZE, high=MAX_HOUSE_SIZE, size=NUM_OF_HOUSES)

    np.random.seed(RANDOM_SEED)
    house_price = house_size * 100.0 + np.random.randint(low=MIN_HOUSE_PRICE, high=MAX_HOUSE_PRICE, size=NUM_OF_HOUSES)
    return house_size, house_price


def __visualise_data(house_size, house_price):
    plt.plot(house_size, house_price, "bx")
    plt.ylabel("Price")
    plt.xlabel("Size")
    plt.show()


def __get_training_data(house_size, house_price):
    train_size = np.asarray(house_size[:NUM_TRAINING_SAMPLES])
    train_price = np.asanyarray(house_price[:NUM_TRAINING_SAMPLES:])

    train_size_norm = __normalize(train_size)
    train_price_norm = __normalize(train_price)

    return train_size, train_size_norm, train_price, train_price_norm


def __get_test_data(house_size, house_price):
    test_size = np.array(house_size[NUM_TRAINING_SAMPLES:])
    test_price = np.array(house_price[NUM_TRAINING_SAMPLES:])

    test_size_norm = __normalize(test_size)
    test_price_norm = __normalize(test_price)

    return test_size, test_size_norm, test_price, test_price_norm
#endregion

#region TF methods
def __configure_placeholders():
    '''
    Placeholders are updated as gradient descent is executing
    '''
    tf_house_size = tf.placeholder("float", name="house_size")  # the name is used in the computation graph
    tf_price = tf.placeholder("float", name="price")            # placeholders are a bit like a pointer

    return tf_house_size, tf_price


def __configure_model_variables():
    '''
    Model variables initialised to random value prior to training
    '''
    tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
    tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

    return tf_size_factor, tf_price_offset


def __configure_model(tf_size_factor, tf_house_size, tf_price_offset):
    '''
    Based on the model of a straight-line y = mx + c: 

        house_price = (size_factor * size) + price_offset

    Note: From reviewing the dataset, we are inferring that a straight-line
    equation would be a good model for this data.

    Other datasets may require a different model
    '''
    return tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)


def __configure_loss_function(tf_price_pred, tf_price):
    '''
    Define the Loss Function (how much error) - Mean squared error
    '''
    return tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2 * NUM_TRAINING_SAMPLES)


def __train_model(cost_fn):
    return tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost_fn)
#endregion

run()


'''
NOTES:
1. The cost decreases to a certain point and then stabilises at a low value
2. size_factor and price_offset also stabilise which is a sign they are trained 
   as well as they can to fit the available data
3. The low cost value means we are fitting the data well
4. The first graph shows the trained model and how well it fits the data

BUGS:
- All the sample dots are being joined up by lines. This shouldn't be the case.
'''