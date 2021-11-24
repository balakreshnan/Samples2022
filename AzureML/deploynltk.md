# Deploy Custom NLTK model to Azure ML inferencing in AKS

## Custom NLTK model deployment

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Create a compute instance for notebook

## Note book Code

```
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.model import Model
```

```
import azureml.core
print(azureml.core.VERSION)
```

```
from azureml.core.workspace import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
```

```
#Register the model
from azureml.core.model import Model
model = Model.register(model_path = "naivebayes.pickle", # this points to a local file
                       model_name = "naivebayes.pickle", # this is the name the model is registered as
                       tags = {'area': "movies ewview", 'type': "nltk"},
                       description = "Movies review Naives classifier",
                       workspace = ws)

print(model.name, model.description, model.version)
```

```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies 

conda_deps = CondaDependencies.create(conda_packages=['numpy','scikit-learn==0.19.1','scipy'], pip_packages=['azureml-defaults', 'inference-schema', 'nltk'])
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps
```

```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies 

conda_deps = CondaDependencies.create(conda_packages=['numpy','scikit-learn==0.19.1','scipy'], pip_packages=['azureml-defaults', 'inference-schema', 'nltk'])
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps
```

```
%%writefile score.py
import os
import pickle
import json
import numpy
import nltk
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace
from azureml.core.model import Model

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'naivebayes.pickle')
    # deserialize the model file back into a sklearn model
    from azureml.core.authentication import ServicePrincipalAuthentication

    svc_pr_password = os.environ.get("AZUREML_PASSWORD")

    svc_pr = ServicePrincipalAuthentication(
        tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        service_principal_password="xxxxxxxxxxxxxxxxxxxxxxxx")


    ws = Workspace(
        subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxx",
        resource_group="mlops",
        workspace_name="mlopsdev",
        auth=svc_pr
        )

    # ws = Workspace.from_config()
    model = Model(ws, 'naivebayes.pickle')
    model.download(target_dir='.', exist_ok=False, exists_ok=None)
    
    classifier_f = open("naivebayes.pickle", "rb")
    model = pickle.load(classifier_f)
    
# This is how the Naive Bayes classifier expects the input
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        #data = numpy.array(data)
        nltk.download('punkt')
        nltk.download('stopwords')
        #result = model.predict(data)
        review_santa = '''

        It would be impossible to sum up all the stuff that sucks about this film, so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume (funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear; an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly and caaarreee-fulll-yyy prrooo-noun-ceeed; a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.

        The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
        '''
        print(review_santa )
        
        words = word_tokenize(review_santa)
        words = create_word_features(words)
        result = model.classify(words)

        #result = nltk.classify.accuracy(model, data))*100
        # you can return any data type as long as it is JSON-serializable
        return result #.tolist()
    except Exception as e:
        error = str(e)
        return error
```

```
from azureml.core.model import InferenceConfig

inf_config = InferenceConfig(entry_script='score.py', environment=myenv)
```

```
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your AKS cluster
aks_name = 'aks-dev' 

# Verify that cluster does not exist already
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # Use the default configuration (can also provide parameters to customize)
    prov_config = AksCompute.provisioning_configuration()

    # Create the cluster
    aks_target = ComputeTarget.create(workspace = ws, 
                                    name = aks_name, 
                                    provisioning_configuration = prov_config)

if aks_target.get_status() != "Succeeded":
    aks_target.wait_for_completion(show_output=True)
```

```
# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration()

# # Enable token auth and disable (key) auth on the webservice
# aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)
```

```
aks_service.delete()
```

```
%%time
aks_service_name ='nltk-service-1'

aks_service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inf_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)
```

```
print(ws.webservices)

# Choose the webservice you are interested in

from azureml.core import Webservice

service = Webservice(ws, 'nltk-service-1')
print(service.get_logs())
```

```
# # if (key) auth is enabled, retrieve the API keys. AML generates two keys.
key1, Key2 = aks_service.get_keys()
print(key1)

# # if token auth is enabled, retrieve the token.
# access_token, refresh_after = aks_service.get_token()
```

```
%%time
import json

test_sample = json.dumps({'data': [
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1]
]})
test_sample = bytes(test_sample,encoding = 'utf8')

prediction = aks_service.run(input_data = test_sample)
print(prediction)
```

```
# https://www.nltk.org/howto/classify.html
```

```
%%time
aks_service.delete()
model.delete()
```
