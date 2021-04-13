import unittest
class Eleve(unittest.TestCase) :
    def __init__(self,age,ID,name) :
        self.ID = ID
        self.name = name
        self.age = age
    def __str__(self) :
        return f'L'Eleve s'appelle {self.name}, d'age {self.age} et son ID est {self.ID}' 
    def getID(self) :
    	return (self.ID)
    def update_ID(self,ID2) :
        self.ID=ID2
