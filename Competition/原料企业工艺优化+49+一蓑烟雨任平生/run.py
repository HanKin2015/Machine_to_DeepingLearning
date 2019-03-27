import os

#os.system('python testB.py')
#os.system('python testC.py')
#execfile('./testB.py')  python3中淘汰
exec(open('./testB.py', encoding = 'utf-8').read())
exec(open('./testC.py', encoding = 'utf-8').read())