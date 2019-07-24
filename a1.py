def main():
    a=input().split()
    n=int(a[0])
    q=int(a[1])
    for i in range(q):
        b=input().split()
        c=input().split()
        for j in range(n):
            b[j]=int(b[j])
            c[j]=int(c[j])
       
        d=input().split()
        d1=int(d[0])
        d2=int(d[1])
        m=0
        for j in range(d1+1,d2+2):
            m=m+(b[j-2]*c[j-2])
        print(m)
main()
            
        