# Project specific
def adjust_y(y):
	'''Adjusting the higher y's to one group'''
	y_adjusted = []
	for score in y:
	    if score>= 31:
	        y_adjusted.append(31)
	    else:
	        y_adjusted.append(score)
	return y_adjusted


#Gini Index
#0 (complete equality) to 1 (complete inequality)
# Rewritten in Python
# From R Code in this link https://www.kaggle.com/wiki/RCodeForGini
def SumModelGini(solution, submission):
  df = pd.DataFrame([solution, submission]).T
  df.columns = ['solution', 'submission']
  df['random'] = np.array(xrange(1,df.shape[0]+1)) / np.array(float(df.shape[0])*len(list(xrange(1,df.shape[0]+1))))
  totalPos = np.sum(df['solution'])
  # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df['cumPosFound'] = np.cumsum(df['solution'])
  # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df['Lorentz'] = df['cumPosFound'] / totalPos
  # will store Lorentz minus random
  df['Gini'] = df['Lorentz'] = df['random']
  print(df)
  return np.sum(df['Gini'])

def NormalizedGini(solution, submission):
  return SumModelGini(solution, submission) / SumModelGini(solution, solution)

