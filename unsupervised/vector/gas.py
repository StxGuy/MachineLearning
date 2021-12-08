import matplotlib.pyplot as pl
import numpy as np

class neuron:
	def __init__(self):
		self.w = np.array([2*(np.random.rand()-0.5), 2*(np.random.rand()-0.5)])
		
	def D(self,x):
		return np.linalg.norm(self.w-x)
		
	def adjust(self,i,x,eta,lbd):
		self.w = self.w + eta*np.exp(-float(i)/lbd)*(x-self.w)

class gas:
	def __init__(self,N,eta):
		self.Nnodes = N
		self.eta = eta
		self.nodes = [neuron() for i in range(N)]
		self.lbd = 0.2
		self.C = -1*np.ones((N,N))
		
	def sort(self,x):
		d = []
		for node in self.nodes:
			d.append(node.D(x))
		self.k = np.argsort(d)
	
	def update(self,x):
		for i in range(self.Nnodes):
			self.nodes[i].adjust(self.k[i],x,self.eta,self.lbd)
		self.eta = self.eta*0.99
	
	def age(self):
		# Connect
		self.C[self.k[0]][self.k[1]] = 0
		self.C[self.k[1]][self.k[0]] = 0
		
		# age
		for i in range(self.Nnodes):
			if (self.C[self.k[0]][i] > -1 and self.k[0] != i):
				self.C[self.k[0]][i] = self.C[self.k[0]][i] + 1
				self.C[i][self.k[0]] = self.C[i][self.k[0]] + 1
				
				# Died
				if (self.C[self.k[0]][i] > 4):
					self.C[self.k[0]][i] = -1
					self.C[i][self.k[0]] = -1
	def adjust(self,x):
		self.sort(x)
		self.update(x)
		self.age()
		
	def show(self):
		for i in range(self.Nnodes-1):
			for j in range(i+1,self.Nnodes):
				if (self.C[i][j] > -1):
					x = [self.nodes[i].w[0],self.nodes[j].w[0]]
					y = [self.nodes[i].w[1],self.nodes[j].w[1]]					
					pl.plot(x,y)
		
		
G = gas(500,0.75)

for it in range(4000):
	# Sample from a disc
	r = 2*(np.random.rand()-0.5)
	t = np.random.rand()*2*np.pi
	x = np.array([r*np.cos(t), r*np.sin(t)])

	G.adjust(x)
	#ax = pl.gca()
	#ax.clear()
	#G.show()
	#pl.pause(0.001)

G.show()

for t in range(360):
	x = np.cos(2*np.pi*float(t)/360)
	y = np.sin(2*np.pi*float(t)/360)
	pl.plot(x,y,'.',color='gray')

pl.show()
