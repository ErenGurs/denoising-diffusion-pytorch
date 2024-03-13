#!/bin/bash

PATH=$PATH:/coreflow/venv/bin/
export $PATH
export $PATH

pip install 'notary-client[conductor]>=2.3.2' --index-url https://pypi.apple.com/simple

alias conductor='AWS_EC2_METADATA_DISABLED=true aws --endpoint-url https://conductor.data.apple.com'
