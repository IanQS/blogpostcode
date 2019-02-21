# Acts as a tester using httpie for our flask app. Easier than setting up the viz

http PUT http://localhost:5000 f='sqrt(4-xs**2)' a:=0 b:=2 c:=0 d:=2
http GET http://localhost:5000
http GET http://localhost:5000/0
