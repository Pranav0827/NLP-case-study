import sys
import types

# Mock scipy.spatial.transform before scipy.spatial can import it
mock_transform = types.ModuleType('scipy.spatial.transform')
mock_transform.RigidTransform = object
mock_transform.Rotation = object
mock_transform.Slerp = object
sys.modules['scipy.spatial.transform'] = mock_transform

import scipy
import scipy.stats
import sklearn
print("sklearn imported successfully!")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print("sklearn modules imported successfully!")
