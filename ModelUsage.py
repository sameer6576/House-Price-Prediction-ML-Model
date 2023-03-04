# coding: utf-8
# In[1]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')


# In[19]:


features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24058048, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:





# In[ ]:




