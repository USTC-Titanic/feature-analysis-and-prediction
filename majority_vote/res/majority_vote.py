## majority vote
import pandas as pd

s1 = pd.read_csv('xmq.csv')
s2 = pd.read_csv('yj.csv')
s3 = pd.read_csv('ljj.csv')
s4 = pd.read_csv('lzh.csv')
s_out = s1

size = len(s_out['Survived'])
for i in range(size):
	v1 = s1['Survived'][i]
	v2 = s2['Survived'][i]
	v3 = s3['Survived'][i]
	sigma = v1 + v2 + v3
	if sigma > 1:
		s_out['Survived'][i] = 1
	else:
		s_out['Survived'][i] = 0

s_out.to_csv('s_out.csv', index=False)


## cnt diff
import pandas as pd
s1 = pd.read_csv('xmq.csv')
s_out = pd.read_csv('s_out.csv')
size = len(s_out['Survived'])
cnt = 0
for i in range(size):
	v1 = s1['Survived'][i]
	v_out = s_out['Survived'][i]
	if v1 != v_out:
		# print(i, v1, v_out)
		cnt += 1

print(cnt)