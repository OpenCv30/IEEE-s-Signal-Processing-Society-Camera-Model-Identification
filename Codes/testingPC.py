import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import pdb,time,random
import os
import numpy as np
import threading 
from queue import Queue
from threading import Lock

def _putInsideQ(thdName,thdId,dataQ,dataQlock):
	batch_data = [ thdName, thdId,thdId*2];
	while True:
		if not dataQ.full():
			print('...........start.............');
			print('Qsize= {}, ThdName {}-Data-{},{}'.format( dataQ.qsize(),thdName,thdId,thdId*2))
			dataQlock.acquire()
			dataQ.put(batch_data);
			dataQlock.release()
			print('Placed  Data, Qsize ={}',dataQ.qsize())
			print('............end.............');
			time.sleep(np.random.random())
		else:
			print('Waiting to get Q empty');
def _readFromQ(thdName,dataQ):
	while True:
		if dataQ.qsize() >0:
			print('.............start............');
			print('Qsize= {}, ThdName {}'.format( dataQ.qsize(),thdName))
			batch = dataQ.get();
			print('Qsize= {},Data-{},{}'.format( dataQ.qsize(),batch[1],batch[2]))
			dataQ.task_done()
			print('Qsize after task_done= {}'.format( dataQ.qsize()))
			print('..........end...............');
			time.sleep(np.random.random())
		else:
			print('Consumes all the data waiting for Q to get fill');

class GenericThread(threading.Thread):
	def __init__(self,name,function,id=0,myargs=None):
		self.threadName = name;
		self.thdId = id
		self.runFunc = function
		self.args = myargs;
		#pdb.set_trace()
		super(GenericThread,self).__init__()
  
	def run(self):
		print(self.threadName + " Starting....")
		#pdb.set_trace()
		if self.args is not None:
			self.runFunc(*self.args)
		else:
			self.runFunc()
		print(self.threadName + " stopping....")
		
	
n_producer =10;
n_consumer =7
dataQ = Queue(25);
dataQlock = Lock();
proDList =[];
consList =[]
for ind in range(n_producer):
	name = 'Thread_Producer_'+str(ind);
	producer_i = GenericThread(name,_putInsideQ,ind,(name,ind,dataQ,dataQlock));
	proDList.append(producer_i);
	
for ind in range(n_consumer):
	name = 'Thread_Consumer_'+str(ind);
	consumer_i = GenericThread(name,_readFromQ,ind,(name,dataQ));
	consList.append(consumer_i);
	
for thd in proDList:
	thd.start();
for thd in consList:
	thd.start();	
	
for thd in proDList:
	thd.join();
for thd in consList:
	thd.join();
	
	 



