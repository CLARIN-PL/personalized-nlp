#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt -f  https://download.pytorch.org/whl/torch_stable.html
