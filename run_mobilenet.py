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
)



def main():
   
    test_runner = AOT_DEFAULT_RUNNER

    mod, params = testing.mobilenet.get_workload(batch_size=1)
    
    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    
    data = np.random.uniform(size=data_shape).astype("float32")
    input_list = {"data": data}
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
