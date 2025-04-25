/// @author Sudhanva Kulkarni
/// Simple implementation of Vector and MXVector objects


#include <cassert>
#include "layouts.h"


namespace Lo_Gemm {

template<typename T, typename idx>
class Vector {
    private :
    idx m;
    T* data;
    idx stride;

    public:
    Vector(T* data, idx m, idx stride = static_cast<idx>(1)) : data(data), m(m), stride(stride) {}

    constexpr inline T& operator[] const (idx i) {
        return data[i*stride];
    }

    constexpr inline T& operator() const (idx i) {
        return data[i*stride];
    }   
    

};

template<typename T, typename idx, typename T_scal>
class MX_Vector {
    private:
    idx m;  //length of data vector
    idx n;  //length of shared_exps vector
    idx stride; //stride of data vector
    idx r1; //number of contiguos elems that share common exp
    T_scal* shared_exps;
    T* data;

    public:
    MX_Vector(T* data, T_scal* shared_exps, idx m, idx n, idx stride = static_cast<idx>(1), idx r1 = static_cast<idx>(1)) : data(data), shared_exps(shared_exps), m(m), n(n), stride(stride), r1(r1) {}

    constexpr inline T& operator[] const (idx i) {
        return data[i*stride];
    }

    constexpr inline T& operator() const (idx i) {
        return data[i*stride] * shared_exps[i/r1];
    }

};

} //namespace Lo_gemm