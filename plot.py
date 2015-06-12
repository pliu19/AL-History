import matplotlib.pyplot as plt

if __name__ == '__main__':
	inlist=[]
	plt.figure()
	with open("unc_Trial_1_proba.csv") as f:
		line = f.readline()
		#while line:
		for i in range(5):
			inlist=map(float,line.split(','))
			plt.subplot(3,3,i+1)
			plt.plot(inlist)
			line = f.readline()
		plt.show()