from PIL import Image
from dnn_app_utils_v2 import *

# set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# make the random matrix same every time
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                       -1).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

# 5-layer model
layers_dims = [12288, 20, 7, 5, 1]


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 10 training example
        if print_cost and i % 10 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# run the training
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

# get the training set accuracy
pred_train = predict(train_x, train_y, parameters)

# get the test set accuracy
pred_test = predict(test_x, test_y, parameters)

# this is the name of the image file
my_image = "my_image.jpg"
# the true class of your image (1 -> cat, 0 -> non-cat)
my_label_y = [1]

num_px = train_x_orig.shape[1]

fname = my_image
image = np.array(plt.imread(fname))
my_image = np.array(Image.fromarray(image).resize(size=(num_px, num_px))).reshape((num_px * num_px * 3, 1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
