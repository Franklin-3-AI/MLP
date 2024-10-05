import math
import numpy as np
import matplotlib.pyplot as plt
import torch

#Creating an Engine for calculating the forward and backward propagation. We are doing it so to know what is exactly going on under the hodd.
class Value:
    def __init__(self,data,_children=(),_op='',label='') :
        self.data =data#a scalar value for waits and inputs
        self._prev= set(_children) #to keep track of cache while backprop
        self._op=_op#this will keep track of the operation in the backprop
        self.label=label
        self.grad=0
        self._backward= lambda:None#to define the backpropagation function

    def __repr__(self):#this will show the self.data in nicer form while we print it
        return f"Value(data={self.data})"
    
    def __add__(self,other):#for addition
        other=other if isinstance(other,Value) else Value(other)# sometimes other might not be the instance of class Value but just a scalar
        out= Value(self.data+other.data,(self,other),'+')
        def _backward():#for the backprop through the '+' node
            self.grad+=1.0*out.grad#backprop through '+' node is just one so the gradient will just pass through and out.grad is due to chain rule
            other.grad+=1.0*out.grad
        out._backward=_backward#Passing up the gradient through '+' node via backward prop
        return out
    
    def __neg__(self):#-self
        return self*-1
    
    def __sub__(self,other): #self - other
        return self + (-other)
    
    def __mul__(self,other):#for multiplication
        other=other if isinstance(other,Value) else Value(other)#sometimes other might not be instance of class Value but just a number
        out= Value(self.data*other.data,(self,other),'*')
        def _backward():#backward prop through * node
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self,other):#defing power so we can define multiplication
        assert isinstance(other,(int,float)),'only supporting int/float powers for now'
        out= Value(self.data**other,(self,),f'**{other}')
        def _backward():#backward prop though '**' node
            self.grad+=out.grad*other*(self.data**(other-1))
        out._backward=_backward
        return out
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other + self
        return other + (-self)
    
    def __rmul__(self,other):#other*self incase we multiply scalar to instance of class Value this will reverse the order, the other case can be handled by __mul__
        return self*other
    
    def __truediv__(self,other):#self/other
        return self*other**-1     
    
    

    
    def tanh(self):#activation function, it doesn't matter how much complicated or non differentiable function you choose as long as you know how to define its derivative precisely to computer
        out= Value(np.tanh(self.data),(self,),'tanh')
        def _backward():
            self.grad+=(1- (out.data)**2)*out.grad
        out._backward=_backward
        return out
    
    def exp(self):#activaton function 'exponential'
        x= self.data
        out = Value(np.exp(x),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out

    
    def backward(self):#this will define the over all backpropation through all the nodes
        topo=[]#using topological sorting to going through all the nodes to get all the derivatives with a particular order
        visited=set()
        def buil_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:#going through each variable by which the v is made up of.
                    buil_topo(child)#calling the function again in the recursive manner, for each variable by which v is made of
                topo.append(v)#appending the node in the topo of to calculate the backprop 
        buil_topo(self)
        self.grad=1.0#as we have setted the self.grad = 0 which we don't want while using chain rule as self.grad for the final output(or loss) with itself will be 1.
        for node in reversed(topo):
            node._backward()

#creating the NN 


class Neuron:#creating a class for a single neuron
    
    def __init__(self,nin): #nin is baiscally number of inputs in the neuron so that we can have weights accordingly
        self.w=[Value(np.random.uniform(-1,1)) for _ in range(nin)]# initializinf the weights in the neuron
        self.b =Value(np.random.uniform(-1,1))
    
    def __call__(self, x):#for doing the multiplication and activation  in the neuron
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)#getting wi*xi+b , zip(x,y) just join the tuple elements like making (x1,x2,x3,...) and (y1,y2,y3,...) to ((x1,y1),(x2,y2),(x3,y3),...)
        out= act.tanh()#activation function
        return out
    
    def parameters(self):#concatenating the parameters for one neuron so that we can call them while gradient descent
        return self.w+[self.b]
    


class Layer:#creating a class for making a layer of neurons

    def __init__(self,nin,nout):#nout will be the number of neurons as they are the output of a single layer
        self.neurons= [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):#concatenating the parameters for on layer so that we can call them while gradient descent
        return [p for neuron in self.neurons for p in neuron.parameters()]# first loop calls one neuron in the layer the second calls the parameters in that neuron
    

    
class MLP: #creating a new class for building up the whole NN i.e. from input to output with multiple layes

    def __init__(self, nin, nouts): #nouts is list containing the number of neurons for every layer that is to be constructed in the whole NN
        sz =[nin]+nouts
        self.layers= [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):# concatenating the parameters for the whole NN
        return [p for layer in self.layers for p in layer.parameters()]# first loop calls a single layer and second calls the parameters in that layer
    
# A simple example
x_train =[[1.0, -2.0, 4.0],
    [2.0, -4.5, 9.0],
    [9.2, -0.2, 0.9],
    [1.1, 2.2, 3.3]]
y_train = [1.0, 1.0, -0.5, 0.5]#target value
Model = MLP(3,[5,6,1])#defining the model

iter=50
for k in range(1000):   
    #forward prop
    ypred = [Model(x) for x in x_train]
    loss = sum((yhat-y)**2 for y,yhat in zip(y_train,ypred)) #defing the square error loss

    #backward prop
    for p in Model.parameters():#setting up the all the gradients of all the parameters to be zero for iteration otherwise they will start to accumulate as we have used += while getting the grads in all the backprop
        p.grad=0.0
    loss.backward()

    #update the parameters in each iteration
    for p in Model.parameters():
        p.data += -0.08 * p.grad

    print(k, loss.data)

