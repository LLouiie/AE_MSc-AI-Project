# Source this before touching WebShop. Not executable on purpose.
#
# WebShop pins 2022-era deps (torch 1.11, numpy 1.22, spacy 3.3, pyserini 0.17)
# and none of them have Python 3.12 wheels, while python3.12 is the only system
# interpreter here and there is no module system. Hence a private Miniconda with
# a 3.8.13 env, built from conda-forge ONLY so that Anaconda's defaults-channel
# Terms of Service never applies.
#
# JAVA_HOME must point at $CONDA_PREFIX/lib/jvm, NOT at $CONDA_PREFIX: pyjnius
# looks for $JAVA_HOME/lib/server/libjvm.so, and conda-forge's openjdk puts the
# JVM one level deeper than the layout pyjnius assumes.

export WEBSHOP_ROOT=/vol/gpudata/jy625-ae-data/webshop_official
export CONDA_PREFIX=/vol/gpudata/jy625-ae-data/miniconda3/envs/webshop
export JAVA_HOME=$CONDA_PREFIX/lib/jvm
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
export PATH=$CONDA_PREFIX/bin:$JAVA_HOME/bin:$PATH
export PYTHON=$CONDA_PREFIX/bin/python
export PIP_CACHE_DIR=/vol/gpudata/jy625-ae-data/pip_cache
export TMPDIR=/vol/gpudata/jy625-ae-data/tmp_ae3_scratch
