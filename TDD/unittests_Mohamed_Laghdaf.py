import unittest
import unittests_Jonathan_Gusching
import Batiment

def main():
    Bat1=Batiment.Batiment(460,"A","Palaisaeu","ResidenceEnsta")
    Bat2=Batiment.Batiment(900,"T","Palaiseau","ResidenceEnsta")
    Bat3=Batiment.Batiment(480,"S","Palaiseau","ResidenceEnsta")
    Bat4=Batiment.Batiment(560,"N","Palaiseau","ResidenceEnsta")
    Bat1.testprice()
    Bat2.testprice()

if __name__ == "__main__":
    main()
