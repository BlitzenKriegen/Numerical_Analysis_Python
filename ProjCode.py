import pandas as pd
import numpy as np
import math
import time

def bisection (f , a , b , Nmax = 1000 , TOL = 1e-4 , Frame = True ) :
    if f (a) * f (b) >= 0:
        print ( " Bisection method is inapplicable . " )
        return None

    # let c_n be a point in ( a_n , b_n )
    an = np.zeros(Nmax , dtype = float)
    bn = np.zeros (Nmax , dtype = float)
    cn = np.zeros (Nmax , dtype = float)
    fcn = np.zeros (Nmax , dtype = float)
    
    # initial values
    an [0] = a
    bn [0] = b
    
    for n in range (0 , Nmax -1) :
        cn [n] = (an[n] + bn [n]) / 2
        fcn [n]= f (cn[n])
        if f (an[n]) * fcn [n] < 0:
            an [n + 1]= an [n]
            bn [n +1]= cn [n]
        elif f(bn[n]) * fcn [n] < 0:
            an [n + 1]= cn[n]
            bn [n + 1]= bn [n]
        else :
            print (" Bisection method fails . ")
            return None
        if ( abs ( fcn [ n ]) < TOL ) :
            if Frame :
                return pd.DataFrame({'an': an [: n +1] , 'bn': bn [: n +1] , 'cn': cn [: n +1] , 'fcn': fcn [: n +1]})
            else:
                return an, bn, cn, fcn, n

    if Frame :
        return pd . DataFrame ({ 'an': an [: n +1] , 'bn': bn [: n +1] , 'cn': cn [: n +1] , 'fcn': fcn [: n
    +1]})
    else :
        return an , bn, cn, fcn, n

def Bisection_step(f, an, bn, cn, fcn, n) :
    cn [n] = (an[n] + bn [n]) / 2
    fcn [n]= f (cn[n])
    if f (an[n]) * fcn [n] < 0:
        an [n + 1]= an [n]
        bn [n + 1]= cn [n]
    elif f(bn[n]) * fcn [n] < 0:
        an [n + 1]= cn[n]
        bn [n + 1]= bn [n]
    else :
        print (" Bisection method fails step. ")
        return None
    return

def find_newBound(f, cn, an, bn, index) :
    aFunc = f(an[index])
    bFunc = f(bn[index])
    cFunc = f(cn[index])
    
    if aFunc * cFunc < 0:
        an [index + 1] = an [index]
        bn [index + 1] = cn [index]
    elif bFunc * cFunc < 0:
        an [index + 1] = cn[index]
        bn [index + 1] = bn [index]
    else :
        print (" Bisection method fails sort. ")
    return

def Bisection_accelerated(f, a, b, TOL, Nmax, Frame = True) :
    an = np.zeros(Nmax , dtype = float)
    bn = np.zeros (Nmax , dtype = float)
    cn = np.zeros (Nmax , dtype = float)
    fcn = np.zeros (Nmax , dtype = float)
    
    an[0] = a
    bn[0] = b
    rg = int(Nmax/3)
    
    for i in range(0,rg):
        el = 3 * i
        Bisection_step(f, an, bn, cn, fcn, el)
        Bisection_step(f, an, bn, cn, fcn, el + 1)
        Aitken_step(cn,el+1)
        find_newBound(f,cn,an,bn, el + 2)
        if(abs(f(cn[el + 2])) < TOL):
            return cn
    return None    
    
def Newton_method (f, Df, x0, TOL, Nmax = 100 , Frame = True ) :
    xn = np.zeros ( Nmax , dtype = float )
    fxn = np.zeros ( Nmax , dtype = float )
    Dfxn = np.zeros( Nmax , dtype = float )
    xn [0] = x0
    for n in range (0 , Nmax -1) :
        fxn[n] = f (xn[n])
        Dfxn [n] = Df(xn[n])
        if abs (fxn[n]) < TOL :
            Dfxn [ n ] = Df ( xn [ n ])
            if Frame :
                return pd . DataFrame ({'xn': xn [: n +1] , 'fxn': fxn [: n +1] , 'Dfxn': Dfxn [: n +1]})
            else :
                return xn , fxn , Dfxn , n
        if Dfxn [n] == 0:
            print ("Zero derivative . No solution found.")
            return None       
        xn [n +1] = xn [ n ] - fxn [ n ]/ Dfxn [ n ]
    print ("Exceeded maximum iterations . No solution found.")
    return None

def Newton_step(f,Df, xn, fxn, Dfxn, index) :
    fxn[index] = f (xn[index])
    Dfxn [index] = Df(xn[index])
    xn [index + 1] = xn [index] - fxn [index]/ Dfxn [index]
    return

def Newton_accelerated(f, Df, x0, TOL, Nmax, Frame = True) :
    xn = np.zeros ( Nmax , dtype = float )
    fxn = np.zeros ( Nmax , dtype = float )
    Dfxn = np.zeros( Nmax , dtype = float )
    xn [0] = x0
    rg = int(Nmax/3)
    for i in range(0,rg):
        el = 3 * i
        Newton_step(f, Df, xn, fxn, Dfxn, el)
        Newton_step(f, Df, xn, fxn, Dfxn, el+1)
        Aitken_step(xn,el+2)
        if(abs(f(xn[el+2])) < TOL):
            return xn
    return None

def Halley_method (f, Df,D2f, x0, TOL, Nmax = 100 , Frame = True ) :
    xn = np.zeros ( Nmax , dtype = float )
    fxn = np.zeros ( Nmax , dtype = float )
    Dfxn = np.zeros( Nmax , dtype = float )
    D2fxn = np.zeros( Nmax , dtype = float )
    xn [0] = x0
    for n in range (0 , Nmax -1) :
        fxn[n] = f (xn[n])
        Dfxn [n] = Df(xn[n])
        D2fxn [n] = D2f(xn[n])
        if abs (fxn[n]) < TOL :
            Dfxn [ n ] = Df ( xn [ n ])
            if Frame :
                return pd . DataFrame ({'xn': xn [: n +1] , 'fxn': fxn [: n +1] , 'Dfxn': Dfxn [: n +1]})
            else :
                return xn , fxn , Dfxn , n
        if Dfxn [n] == 0:
            print ("Zero derivative . No solution found.")
            return None       
        xn [ n +1] = xn [ n ] - (2*fxn[n]*Dfxn[n])/(2*Dfxn[n]**2 - fxn[n]*D2fxn[n])
    print ("Exceeded maximum iterations . No solution found.")
    return None

def Halley_step(f,Df,D2f, xn, fxn, Dfxn, D2fxn, index) :
    fxn[index] = f (xn[index])
    Dfxn [index] = Df(xn[index])
    D2fxn [index] = D2f(xn[index])
    xn [index + 1] = xn[index] - (2*fxn[index]*Dfxn[index])/(2*Dfxn[index]**2 - fxn[index]*D2fxn[index])
    return

def Halley_accelerated(f, Df, D2f, x0, TOL, Nmax, Frame = True) :
    xn = np.zeros ( Nmax , dtype = float )
    fxn = np.zeros ( Nmax , dtype = float )
    Dfxn = np.zeros( Nmax , dtype = float )
    D2fxn = np.zeros( Nmax , dtype = float )
    xn [0] = x0
    rg = int(Nmax/3)
    for i in range(0,rg):
        el = 3 * i
        Halley_step(f, Df, D2f, xn, fxn, Dfxn, D2fxn, el)
        Halley_step(f, Df, D2f, xn, fxn, Dfxn, D2fxn, el+1)
        Aitken_step(xn,el+2)
        if(abs(f(xn[el+2])) < TOL):
            return xn
    return None
    
def Aitken_step(xn, index):
    num = (xn[index-1] - xn[index-2])**2
    den = xn[index] - 2*xn[index-1] + xn[index-2]
    xn [index + 1] = xn[index-2] - num/den
    return

f = lambda x : (1/3)*(2 - math.exp(x) + x**2)
Df = lambda x : (1/3)*(2*x - math.exp(x))
D2f = lambda x: (1/3)*(2 - math.exp(x))

g = lambda x : x**3 - x - 2
Dg = lambda x: 3*(x**2) - 1
D2g = lambda x: 6*x

h = lambda x: x**4 - x**3 + (1/2)*x**2 - 1
Dh = lambda x: 4*x**3 - 3*x**2 + x
D2h = lambda x: 12*x**2 - 6*x

'''
start = time.time_ns()
bisection = bisection(f, a = 1, b = 2, Nmax = 25, TOL = 1e-8)
end = time.time_ns()
print("Time took:", end-start)
print(bisection)

start = time.time_ns()
halley = Halley_method(f,Df,D2f,x0 = 1,TOL = 1e-8 ,Nmax = 20)
end = time.time_ns()
print("Time took:", end-start)
print(halley)

start = time.time_ns()
newton = Newton_method(f,Df, 1, 1e-8, 20)
end = time.time_ns()
print("Time took:", end-start)
print(newton)

start = time.time_ns()
nAcc = Newton_accelerated(h,Dh,1,1e-8,20)
end = time.time_ns()
for i in nAcc:
    if(i != 0) :
        print(i)
print("Time took:", end-start)
        
start = time.time_ns()
hAcc = Halley_accelerated(f,Df, D2f, 1, 1e-8,20)
end = time.time_ns()
for i in hAcc:
    if(i != 0) :
        print(i)
print("Time took:", end-start)

start = time.time_ns()
bAcc = Bisection_accelerated(f,Df, D2f, 1, 1e-8,20)
end = time.time_ns()
for i in bAcc:
    if(i != 0) :
        print(i)
print("Time took:", end-start)
'''