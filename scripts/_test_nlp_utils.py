import nlp_utils
print('scoreDataFrame' in dir(nlp_utils))
print([n for n in dir(nlp_utils) if 'score' in n.lower()])
print('done')
