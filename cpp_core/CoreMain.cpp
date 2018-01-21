#include <iostream>
#include <glog/logging.h>
#include "CoreGoBoard.h"

using namespace std;
using namespace corego;

int main(int argc, char * argv[]){
  int result = 0;
  
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  LOG(INFO) << "INFO:"<<"Starting the test of CoreGoBoard" << endl;

  GoBoard goBoard;

  LOG(INFO)  << goBoard.getDebugString() << endl;

  LOG(INFO) << "Board size:" << goBoard.boardSize << endl;
  
  google::ShutdownGoogleLogging();

  return 0;
}

