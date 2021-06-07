#!/usr/bin/env python3

import os.path
import shutil

print('\nCAMAP installed? Last step! Copying training data to ~/.CAMAP')
data_dir = os.path.normpath(os.path.expanduser('~/.CAMAP'))
os.makedirs(os.path.join(data_dir, 'training_datasets'), exist_ok=True)
shutil.rmtree(os.path.join(data_dir, 'training_datasets'))
shutil.copytree('./training_datasets', os.path.join(data_dir, 'training_datasets'))
print('All done.')
