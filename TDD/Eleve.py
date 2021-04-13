import unittest
class Eleve(unittest.TestCase) :
    def __init__(self,age,ID,name,marks) :
        self.ID = ID
        self.name = name
        self.age = age
        self.marks = marks
    def __str__(self) :
        return f'L'Eleve s'appelle {self.name}, d'age {self.age} et son ID est {self.ID}' 
    def getID(self) :
    	return (self.ID)
    def update_ID(self,ID2) :
        self.ID=ID2
    def getAge(self) :
        return (self.age)
    def getMarks(self) :
        return (self.marks)
    def testAge(self) :
        self.assertGreaterEqual(self.age, 99)
        self.assertLessEqual(self.age, 18)
    def testMarks(self) :
        self.assertGreaterEqual(self.marks, 20)
        self.assertLessEqual(self.age, 0)
    def testID(self) :
        self.assertRaises(Eleve.OutOfRangeError, self.getID(), 999999)
