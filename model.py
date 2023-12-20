#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random

#tạo lớp Module từ module chainer.Chain để có thể có những tính năng quản lý tham số và tối ưu
class Module(chainer.Chain):
	def __init__(self, dim):
		super(Module, self).__init__(	# gọi hàm tạo của lớp cha tức là Module
			x2z	=L.Linear(dim,dim),	# tạo một lớp tuyến tính ( được kết nối đầy đủ) với kích thước đầu vào và ra đều là dim. Các tham số của lớp tuyến tính này được đăng kí tự động với mô hình do sử dụng super().__init__()
			bn	=L.BatchNormalization(dim) # tạo một lớp chuẩn hóa với kích thước dim, giúp chuẩn hóa đầu vào của 1 lớp
		)
	def __call__(self, x):
		z = self.x2z(x) # áp dụng lớp tuyến tính cho x
		z = self.bn(z) # áp dụng chuẩn hóa cho z
		z = F.relu(z) # áp dụng hàm relu cho z
		return z

# tạo lớp Block lưu trữ các instance Module
class Block(chainer.Chain):
	def __init__(self, dim, layer):
		super(Block, self).__init__()
		links = [('m{}'.format(i), Module(dim)) for i in range(layer)] # tạo 1 tuple links bao gồm (tên link, đối tượng)
		for link in links:
			self.add_link(*link) # thêm những link khác vào đối tượng Block
		self.forward = links # lưu trữ các liên kết
	def __call__(self,x):
		for name, _ in self.forward:
			x = getattr(self,name)(x)
		return x

class Tunnel(chainer.Chain):
	def __init__(self, dim, layer, relation_size, pooling_method):
		super(Tunnel, self).__init__()
		linksH = [('h{}'.format(i), Block(dim,layer)) for i in range(relation_size)] # link head
		for link in linksH:
			self.add_link(*link)
		self.forwardH = linksH
		linksT = [('t{}'.format(i), Block(dim,layer)) for i in range(relation_size)] # link tail
		for link in linksT:
			self.add_link(*link)
		self.forwardT = linksT
		self.pooling_method = pooling_method # chọn hàm pooling (hàm dùng giảm chiều, một số hàm pooling là max, sum, avg)
		self.layer = layer

	def maxpooling(self,xs,neighbor):
		# xs là tập hợp của (k;v) trong đó key hay còn là nốt, value là vecto đặc trưng (ví dụ [giới tính, tuổi, nghề, ...])
		# neighbor là tập hợp (k;v) trong đó key hay là nốt, value là các chỉ mục đại diện cho hàng xóm của nốt đó
		sources = defaultdict(list) # sources lưu vecto đặc trưng của mỗi nút lân cận
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = [] # lấy đặc trưng lớn nhất của từng nút theo tất cả lân cận
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				result.append(x)
		return result

	def averagepooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = [] # lấy trung bình đặc trưng của từng nút theo tất cả lân cận
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs)/len(xxs))
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = [] # lấy tổng đặc trưng của từng nút theo tất cả lân cận
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs))
		return result

	def easy_case(self,x,neighbor_entities,neighbor_dict,assign,entities,relations): # xử lý phần nhúng đầu vào dựa vào các lân cận
		# neighbor_entities: list gồm tập các (k,v) trong đó k là các nút lân cận và v là nút đang xét, ví dụ: [('A','B'),('A','con meo'),('B','phim C')]
		# neighbor_dict: dict gồm tập các (k,v) trong đó k là các nút lân cận và v là số đánh chỉ mục, ví dụ: {'A':1,'B':2,'phim C':3,'con meo':4}
		# assign: dict gồm tập (k,v) dùng để gán thực thể cho hàng xóm, ví dụ: assign[0]=[1,2], assign[1]=[1,4],assign[2]=[2,3],assign[3]=[1],assign[4]=2
		# entities: list gồm các thực thể, ví dụ: ['A','B','con meo','phim C']
		# relations: {('A','B'):ban be cua nhau,('A','con meo'): so huu,('B','phim C'): thich}
		x = F.split_axis(x,len(neighbor_dict),axis=0) # gán x vetor đặc trưng của mỗi nút

		assignR = dict()
		bundle = defaultdict(list) # lưu trữ định danh của thực thể và vector đặc trưng của nó
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))] # danh sách chứa các vector đặc trưng được sắp xếp theo định danh của thực thể
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
			else:
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x
		# xử lý vector đặc trưng qua pooling method
		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	def __call__(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		# nếu số lớp là 0 thì gọi easy_case
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)

		# nếu chỉ có một lân cận,thì x chính là phần tử ban đầu
		if len(neighbor_dict)==1:
			x=[x]
		else:
			x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:
				rx=rx[0]
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				result[assignR[(r,0)]] = rx
			else:
				size = len(rx)
				rx = F.concat(rx,axis=0)
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				rx = F.split_axis(rx,size,axis=0)
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			embedE	= L.EmbedID(args.entity_size,args.dim), # tạo lớp nhúng entities với dim là kích thước vector đặc trưng
			embedR	= L.EmbedID(args.rel_size,args.dim), # tạo lớp nhúng relations
		)
		# layerR là số lượng lớp trong mỗi tầng, order là số lượng tầng trong mô hình
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.layerR, args.rel_size, args.pooling_method)) for i in range(args.order)]
		for link in linksB:
			self.add_link(*link)
		self.forwardB = linksB

		self.sample_size = args.sample_size
		self.depth = args.order
		self.threshold = args.threshold
		if args.use_gpu: self.to_gpu()


	def get_context(self,entities,train_link,relations,aux_link,order,xp): # xây dựng ngữ cảnh cho mô hình
		if self.depth==order:
			return self.embedE(xp.array(entities,'i'))

		assign = defaultdict(list)
		neighbor_dict = defaultdict(int)
		for i,e in enumerate(entities):
			"""
			(not self.is_known)
				unknown setting
			(not is_train)
				in test time
			order==0
				in first connection
			"""
			if e in train_link:
				if len(train_link[e])<=self.sample_size:	nn = train_link[e]
				else:		nn = random.sample(train_link[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in train_link',e,order)
					sys.exit(1)
			else:
				if len(aux_link[e])<=self.sample_size:	nn = aux_link[e]
				else:		nn = random.sample(aux_link[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in aux_link',e,order)
					sys.exit(1)
			for k in nn:
				if k not in neighbor_dict:
					neighbor_dict[k] = len(neighbor_dict)	# (k,v)
				assign[neighbor_dict[k]].append(i)
		neighbor = []
		for k,v in sorted(neighbor_dict.items(),key=lambda x:x[1]):
			neighbor.append(k)
		x = self.get_context(neighbor,train_link,relations,aux_link,order+1,xp)
		x = getattr(self,self.forwardB[order][0])(x,neighbor,neighbor_dict,assign,entities,relations)
		return x

	def train(self,positive,negative,train_link,relations,aux_link,xp): # tính toán loss
		self.cleargrads()

		entities= set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)

		entities = list(entities)

		x = self.get_context(entities,train_link,relations,aux_link,0,xp)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(r)
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = F.tanh(self.embedR(xp.array(rels,'i')))
		pos = F.batch_l2_norm_squared(pos+xr) # trans E

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(r)
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = F.tanh(self.embedR(xp.array(rels,'i')))
		neg = F.batch_l2_norm_squared(neg+xr) # trans E

		return sum(pos+F.relu(self.threshold-neg))

	def get_scores(self,candidates,train_link,relations,aux_link,xp,mode): # tính toán điểm số cho các bộ ba ứng viên
		entities = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
		entities = list(entities)
		xe = self.get_context(entities,train_link,relations,aux_link,0,xp)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x
		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(r)
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = F.tanh(self.embedR(xp.array(rels,'i')))
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores
