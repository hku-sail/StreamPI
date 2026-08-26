#! /bin/bash

export CALVIN_ROOT=${CALVIN_ROOT:projects/calvin}

export EGL_LIBS_DIR=$PWD/tools/egl/lib
export LD_LIBRARY_PATH=$EGL_LIBS_DIR:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=$EGL_LIBS_DIR/10_nvidia.json

export PYTHONPATH=$PYTHONPATH:$PWD

python examples/calvin/main.py \
  --calvin_root $CALVIN_ROOT \
  --save_name pi05_calvin_stream5 \
  --host 0.0.0.0 \
  --port 8000
