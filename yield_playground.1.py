
def function_with_yield():
  for i in range(10):
    yield i

def testing():

  a = function_with_yield()
  for i in range(100):
    print (a.next())
  
  

testing()