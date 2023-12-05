import os, sys
import numpy as np
import pandas as pd
from pprint import pprint
sys.path.append('scripts')

from rdkit.Chem import PandasTools
PandasTools.RenderImagesInAllDataFrames(images=True)

from Synthesis import init_LocalTransform, predict_product

dataset = 'USPTO_480k' # get the info of derived templates
scenario = 'mix' # 'sep' or 'mix'

device = 'cpu' # cpu or cuda
model_name = 'LocalTransform_%s' % scenario
model_path = 'models/%s.pth' % model_name
config_path = 'data/configs/default_config'
data_dir = 'data/%s' % dataset

args = {'data_dir': data_dir, 'model_path': model_path, 'config_path': config_path, 'device': device, 'mode': 'test'}
model, graph_functions, template_dicts, template_infos = init_LocalTransform(args)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('# model parameters: %.2fM' % (params/1000000))

reactants = 'N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.C1CCOC1.[H-].[Na+]'
reactants = 'N[C@H]1CNC(=O)C1.Cn1cc(-c2cccc(CCl)c2)cn1'
sep = False
verbose = 1
results_df, results_dict = predict_product(args, reactants, model, graph_functions, template_dicts, template_infos, verbose = verbose, sep = sep)
pprint (results_dict)
results_df
