#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# In[ ]:


DATA_FOLDER = 'dataset/'

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
# train_df = pd.DataFrame(train_data)

train_df = pd.read_csv(DATA_FOLDER + "train.csv")
# Preview the first 5 lines of the loaded data
# train_data.head()
# print(type(train_data))

val_df = pd.read_csv(DATA_FOLDER + "val.csv")



# In[ ]:


# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', args={'fp16': True, 'reprocess_input_data': False, 'train_batch_size': 64, 'num_train_epochs': 1,'eval_batch_size': 64}) # You can set class weights by using the optional weight argument


# In[ ]:


# Train the model
model.train_model(train_df)


# In[ ]:


# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_df)

