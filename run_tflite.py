# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tensorflow as tf
import os
import pytest

pytest.importorskip("tflite")
pytest.importorskip("tensorflow")

from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from backend import VanillaAcceleratorBackend
from tvm.relay import transform, testing
from collections import OrderedDict
import numpy as np


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
    create_relay_module_and_inputs_from_tflite_file,
)

def generate_tflite_file(tflite_filename):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    tf_model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    output = tf_model(x_train[:1])
    output = output.numpy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss(y_train[:1], output).numpy()
    tf_model.compile(metrics=["accuracy"], optimizer="adam", loss=loss)
    tf_model.fit(x_train, y_train, epochs=1)

    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = tflite_converter.convert()
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)

def main():
   
    test_runner = AOT_DEFAULT_RUNNER

    tflite_file = "/tmp/model.tflite"
    if os.path.exists(tflite_file):
        os.remove(tflite_file)
    generate_tflite_file(tflite_file)

    pytest.importorskip("tflite")

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    tf_model_details = interpreter.get_input_details()
    mod, _, params = create_relay_module_and_inputs_from_tflite_file(
        tflite_file, bind_params_by_name=False
    )


    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()

    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    # Generation of test input and output
    data_shape = [int(x) for x in mod["main"].params[0].type_annotation.shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    input_list = {str(tf_model_details[0]["name"]): data}
    output_list = generate_ref_data(mod, input_list, params)
   
    mod = uma_backend.partition(mod)

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")

    compile_and_run(
        AOTModel(module=mod, inputs=input_list, outputs=output_list, params=params),
        test_runner,
        interface_api="c",
        use_unpacked_api=True,
        workspace_byte_alignment=1,
        debug_calculated_workspaces=False,
        target=[target_c, target],
        test_dir=str(export_directory),
    )


if __name__ == "__main__":
    main()
