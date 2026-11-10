from timeit import default_timer as timer
from numba import jit

@jit(nopython=True) # 
def func2(max,obj):
    x = 0
    y = 0
    z = 0 
    x3 = 0
    y3 = 0
    z3 = 0
    conta = 0 # contador de interações 
    for line in range(max):       
        x = x - 1 
        x3 = x * x * x         
        y = 0
        #print (x)        
        for line2 in range(max):                  
            y = y + 1
            y3 = y * y * y 
            resp = x3 + y3             
            # se for maior que objetivo, sai desse loop
            if (resp > obj):                              
                break             
            dif = -(x)-y  # aumenta performance                 
            z = dif # inicia proximo loop já com valor da diferença
            #print (y)
            for line3 in range(max):
                z = z + 1        
                z3 = z * z * z
                resp = x3 + y3 + z3
                # se for maior que objetivo, sai desse loop
                if ( resp > obj): # aumenta performance de 222 Seg pra 6 seg
                    break
                conta = conta + 1
                if (resp == obj): # com if demora muito (obj=26,33,42 dificil)
                    solu = "\n Contador: " + str(conta) +" Numeros: " + str(x) +" , "+ str(y)+" , "+ str(z) 
                    print (solu)
    return solu
                                    
if __name__ == "__main__":
    max = 3000
    obj = 26
    resp = timer()
    print ("\nIniciando ...")
    print ("\n Números encontrados:" ,func2(max,obj))
    print("\nFim Timer:", timer() - resp)
