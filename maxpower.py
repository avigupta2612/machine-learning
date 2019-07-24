def main():
    a=int(input())
    b=int(input())
    c=1
    d=0
    if b%10==1:
        print('0')
    else:
        while c<b/2:
            c=f(c)
            if b%c==0:
                d=d+1
        print(d)
def f(c):
    c=c*2
    return c
main()