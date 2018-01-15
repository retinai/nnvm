import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list
import keras

# prevent keras from using up all gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def verify_keras_frontend(keras_model):
    in_shape = [dim.value if dim.value is not None else 1 for dim in keras_model.input_layers[0].input.shape]
    out_shape = [dim.value if dim.value is not None else 1 for dim in keras_model.output_layers[0].output.shape]

    def get_keras_output(x, dtype='float32'):
        return keras_model.predict(x)

    def get_tvm_output(x, target, ctx, input_name='data', dtype='float32'):
        sym, params = nnvm.frontend.from_keras(keras_model)
        shape_dict = {input_name : x.shape}
        with nnvm.compiler.build_config(opt_level=2):
            graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    x = np.random.uniform(size=in_shape)
    keras_out = get_keras_output(x)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(x.transpose([0,3,1,2]), target, ctx)
        np.testing.assert_allclose(keras_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_forward_softrelu():
    print("test_forward_softrelu")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Activation('softplus')(data)
    x = keras.layers.Concatenate()([x, x])
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_leaky_relu():
    print("test_forward_leaky_relu")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.LeakyReLU(alpha=0.3)(data)
    x = keras.layers.Add()([x, x])
    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_dense():
    print("test_forward_dense")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(data)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='relu', kernel_initializer='uniform')(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_even():
    print("test_forward_conv_even")
    data = keras.layers.Input(shape=(32,32,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,2), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(2,3), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,5), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(5,6), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,2), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(2,3), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(5,5), padding='valid')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_odd():
    print("test_forward_conv_odd")
    data = keras.layers.Input(shape=(31,31,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,2), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(2,3), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,4), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(5,6), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,1), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(1,2), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(2,3), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(5,5), padding='valid')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_even_small():
    print("test_forward_conv_even_small")
    data = keras.layers.Input(shape=(4,4,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), padding='same')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_odd_small():
    print("test_forward_conv_odd_small")
    data = keras.layers.Input(shape=(5,5,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(5,5), padding='same')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_even_strided():
    print("test_forward_conv_even_strided")
    data = keras.layers.Input(shape=(32,32,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(2, 1), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), strides=(1, 2), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(1, 2), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), strides=(2, 1), padding='valid')(x)
    print(keras.models.Model(data, x).output_shape)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_conv_odd_strided():
    print("test_forward_conv_odd_strided")
    data = keras.layers.Input(shape=(31,31,4))
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(1, 2), padding='same')(data)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), strides=(2, 1), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(1, 2), padding='valid')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(4,4), strides=(2, 1), padding='valid')(x)
    print(keras.models.Model(data, x).output_shape)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_transpose_conv():
    print("test_forward_transpose_conv")
    data = keras.layers.Input(shape=(32,32,4))
    x = data
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(2,2), padding='same')(data)
    x = keras.layers.Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(2, 2), padding='same')(x)
    x = keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(2,2), padding='valid')(data)
    x = keras.layers.Conv2DTranspose(filters=10, kernel_size=(3,3), strides=(2, 2), padding='valid')(x)
    x = keras.applications.mobilenet.DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), padding='valid')(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2, 2), padding='valid')(x)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2, 2), padding='same')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_separable_conv():
    print("test_forward_separable_conv")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.SeparableConv2D(filters=10, kernel_size=(3,3),
        padding='same', activation='relu')(data)
    x = keras.layers.BatchNormalization(scale=True, center=False,
        beta_initializer='uniform', gamma_initializer='uniform')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_upsample():
    print("test_forward_upsample")
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.UpSampling2D(size=(3,3))(data)
    x = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)


def test_forward_vgg16():
    print("test_forward_vgg16")
    keras_model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_xception():
    print("test_forward_xception")
    keras_model = keras.applications.xception.Xception(include_top=True, weights=None,
        input_shape=(299,299,3), classes=1000)
    verify_keras_frontend(keras_model)


def test_forward_resnet50():
    print("test_forward_resnet50")
    keras_model = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
        input_shape=(224,224,3), classes=1000)
    verify_keras_frontend(keras_model)


if __name__ == '__main__':
    test_forward_softrelu()
    test_forward_leaky_relu()
    test_forward_dense()
    test_forward_conv_even()
    test_forward_conv_odd()
    test_forward_conv_even_small()
    test_forward_conv_odd_small()
    test_forward_conv_even_strided()
    test_forward_conv_odd_strided()
    test_forward_transpose_conv()
    test_forward_separable_conv()
    test_forward_upsample()

    test_forward_vgg16()
    test_forward_xception()
    test_forward_resnet50()
