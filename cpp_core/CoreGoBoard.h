#ifndef THETA_GO_CORE_GO_BOARD
#define THETA_GO_CORE_GO_BOARD


#include <iostream>
#include <string>

using namespace std;

namespace corego{

  class GoBoard{

    public:
      GoBoard();
      
      static const int boardSize = 19;
      static const int historyLength = 1024;

      string getDebugString();
  
      
    private:

      int board[historyLength][boardSize][boardSize]; 
      
    
  };
}

#endif