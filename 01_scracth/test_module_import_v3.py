import os
import sys
import importlib

path = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_repo\\04_models\\02_single_process\\model_attention_v4\\model_attn_v4.py'

module_name = os.path.splitext(os.path.basename(path))[0]
module_path = path

spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)   


#%%
import os
import importlib.machinery

path = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_repo\\04_models\\02_single_process\\model_attention_v4\\model_attn_v4.py'

module_name = os.path.splitext(os.path.basename(path))[0]
module_path = path

loader = importlib.machinery.SourceFileLoader(module_name, module_path)
mod = loader.load_module()
