#include "Matrix.h"
#include "Vector.h"

namespace LoGemm {

    typedef struct {
        int m;
        int n; 
        int k;
        int ldA, ldB, ldC;
        int 
    } GemmArgs

    template<typename ElementA, typename LayoutA, 
        typename ElementB, typename LayoutB,
        typename ElementC, typename LayoutC,
        typename ElementAc1, typename LayoutAc1,
        typename ElementAc2, typeanme LayoutAc2,
        typename ElementAc3, typeanme LayoutAc3>
        class Gemm {

        }

}