# Makefile of cpp program
#

CC = g++
INCLUDES:=.

all: core_all


core_all: core_build
	../bin/core_goboard

core_build: *.cpp *.h
	$(CC) -lglog -lcurl -Iinclude *.cpp -o ../bin/core_goboard