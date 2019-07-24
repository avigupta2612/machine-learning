import math

def main():
    a=int(input())
    for i in range(a):
        b=int(input())
        count=0
        while b!=0:
            c=issqrt(b)
        
            b=b-(c*c)
            
            count=count+1
        print(count)
def issqrt(b):
    a=math.sqrt(b)
    a=int(a)
    return a
main()