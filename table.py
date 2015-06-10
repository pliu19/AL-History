def work_table(inlist):
	result=[]
	lastpred=inlist[0]
	length=len(inlist)
	lastcross=0
	count=0
	for i in range(length-1):
		if inlist[i]!=lastpred:
			count+=1
		lastcross=i
		lastpred=inlist[i]
	if inlist[length-2]==inlist[length-1]:
		correct=1
	else:
		correct=0
	result.append(count)
	result.append(lastcross)
	result.append(correct)
	return result

if __name__ == '__main__':
	pred=[]
	result=[]
	with open("unc_Trial_1_prediction.csv") as f:
		line = f.readline()
		while line:
			pred=map(int,map(float,line.split(',')))
			result.append(work_table(pred))
			line = f.readline()
	print result