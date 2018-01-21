#include "CoreGoBoard.h"
#include <glog/logging.h>
#include <curl/curl.h>
#include <time.h>
#include <string>

using namespace std;


namespace corego{
  
  GoBoard::GoBoard(){

  }

  string GoBoard::getDebugString(){
    stringstream result;
    
    result << "GoBoard:" << endl;

    result << this->board[0][0][0] << endl;

    

    return result.str();

  }


}