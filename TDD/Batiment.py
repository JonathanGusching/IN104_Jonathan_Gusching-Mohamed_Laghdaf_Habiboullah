class Batiment :
	def __init__(self,price,role,place,name) :
		self.role = role
		self.name=name
		self.__price = price
		self.place=place

	def __price(self) :
		print(self.price)
    	def __str__(self):
        	return f'Le batiment est {self.name}, de type {self.role} qui vaut {self.price} situé {self.place}'
	def getRole(self) :
		return (self.role)
	def setRole(self,new_role)
		self.role=new_role
	def getPlace(self) :
		return (self.place)
	def setPlace(self, new_place)
		self.place=new_place
	def getName(self)
		return self.name
	def rename(self,new_name)
		self.name=new_name
