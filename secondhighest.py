def main():

    a=int(input())
    b=input().split()
    b=map(lambda x:int(x),b)
    c=set(b)
    d=[]
    d=map(lambda y:y,c)
    e=sorted(d)
    
    print(e[len(e)-2])

main()

