network.exe: main.o network.o
	g++ -o network.exe main.o network.o
main.o:main.cpp network.hpp
	g++ -c main.cpp
network.o:network.cpp network.hpp
	g++ -c network.cpp
clean:
	rm *.exe*.o*.stackdump *~
