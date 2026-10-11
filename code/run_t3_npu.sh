#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/torch_npu/lib:$HOME/.local/lib/python3.12/site-packages/torch/lib:$HOME/.local/lib/python3.12/site-packages/vendors/customize/op_api/lib:$ASCEND_HOME_PATH/x86_64-linux/lib64:$LD_LIBRARY_PATH"
export ASCEND_CUSTOM_OPP_PATH="$HOME/.local/lib/python3.12/site-packages/vendors/customize"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec python3 /home/zbw/nlp/HW-1/t3_compare.py