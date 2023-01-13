#include <iostream>
#include <fstream>
#include <string>
#include "gurobi_c++.h"
#include <sstream>

using namespace std;

/*read data*/
int main()
{
/*build environment*/
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        /* variable setting*/
        //
        GRBVar A;
        A = model.addVar(5000, GRB_INFINITY, 0, GRB_CONTINUOUS);
        GRBVar B;
        B = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
        GRBVar C;
        C = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
        GRBVar D;
        D = model.addVar(4000, GRB_INFINITY, 0, GRB_CONTINUOUS);
        
        model.update();

        /*Objectives*/
        GRBLinExpr sum = 0,sum1=0,sum2=0,sum3=0;
        sum = (350-255)*A + (550-320)*B + (450-356)*C + (700-465)*D;
        sum1 = 100*A + 150*B + 200*C + 250*D;
        model.setObjective(sum-sum1, GRB_MAXIMIZE);
        /*Constraints*/
        //1
        sum1 = 0.05*A + 0.05*B + 0.1 *C + 0.15 *D;
        model.addConstr( sum1 <= 1000);
        //2
        sum2 = 0.1*A + 0.15*B + 0.2 *C + 0.05 *D;
        model.addConstr( sum2 <= 2000);
        //3
        sum3 = 0.05*A + 0.1 *B + 0.1 *C + 0.15 *D;
        model.addConstr( sum3 <= 1500);
       
        /*Solve the problem*/
        model.update();
        model.optimize();

        /*Output the result*/
        cout << "\nObj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        
        cout<<"\nA:" <<endl;
        cout<<A.get(GRB_DoubleAttr_X)<<endl;
        cout<<"\nB:" <<endl;
        cout<<B.get(GRB_DoubleAttr_X)<<endl;
        cout<<"\nC:" <<endl;
        cout<<C.get(GRB_DoubleAttr_X)<<endl;
        cout<<"\nD:" <<endl;
        cout<<D.get(GRB_DoubleAttr_X)<<endl;
 
        
           
           
    }
    catch (GRBException message) {
        cout << "Error code = " << message.getErrorCode() << endl;
        cout << message.getMessage() << endl;
    }
    catch (...) {
        cout << "Exception during optimization" << endl;
    }
    system("pause");
    return 0;
}
