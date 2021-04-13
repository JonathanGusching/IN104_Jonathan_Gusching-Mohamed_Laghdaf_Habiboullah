import unittest
import unittests_Mohamed_Laghdaf
import unittests_Jonathan_Gusching
import Eleve

def main():
    eleve1=Eleve.Eleve(20,264157,"Pierre Dupont",16.3)
    eleve2=Eleve.Eleve(-1,264157,"Jeanne Dupont",16.3)
    eleve3=Eleve.Eleve(18,2641575,"Jeanne d'Arc",10.9)
    eleve4=Eleve.Eleve(18,26415,"Quelqu'un",22)
    eleve1.testAge()
    eleve1.testID()
    eleve1.testMarks()
    eleve2.testAge()
    eleve3.testID()
    eleve4.testMarks()

if __name__ == "__main__":
    main()
