import os

IACcorpus_dir = 'IACcorpus'

corpus_path = os.path.join(IACcorpus_dir, os.listdir(IACcorpus_dir)[0])
with open(corpus_path) as f:
    lines = f.readlines()


iac_dict = {}
lines = lines[:30]
line = lines[15]

iacs = line.split('##')[0].split(', ')
iac = iacs[0]
implicit = iac.split('[')
line
lines[15]
