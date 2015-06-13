import matplotlib.pyplot as plt

if __name__ == '__main__':
	inlist = []
	count = 1
	datasets = 30
	plt.figure(figsize=(96,16))	
	for i in range(10):
		with open("unc_Trial_"+str(i+1)+"_proba.csv") as f:
			line = f.readline()
		#while line:
			for j in range(datasets):
				inlist=map(float,line.split(','))
				true_label=1-inlist.pop()
				ax = plt.subplot(10,datasets,count)
				ax.xaxis.set_major_locator(plt.NullLocator())
				ax.yaxis.set_major_locator(plt.NullLocator())
				plt.plot(inlist)
				plt.plot(len(inlist)-1,true_label,marker="o",markersize=15,alpha=0.3)
				line = f.readline()
				count+=1
	#plt.show()
	plt.savefig("/Users/Dany/Desktop/1.png")