#!/usr/bin/env python
# coding: utf-8

# In[1]:


from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# In[2]:


DATA_FOLDER = 'dataset/'

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
test_df = pd.read_csv(DATA_FOLDER + "test.csv")


# In[9]:


# Create a ClassificationModel
model = ClassificationModel('roberta', 'outputs/', args={'fp16': True, 'reprocess_input_data': False, 'train_batch_size': 64, 'num_train_epochs': 1,'eval_batch_size': 64})

all_texts = test_df.values.flatten().tolist()
predictions, _ = model.predict(all_texts)
list_label = predictions

test_df['label'] = list_label
test_df.to_csv('submission.csv', index=False)

