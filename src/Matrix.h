/// @author Sudhanva Kulkarni
/// Simplke implementation of Varying Layout dense matrix


#include <cassert>
#include "layouts.h"


namespace Lo_Gemm {

template<typename T, typename idx, Layout L = ColMajor>
class Matrix {
    
    idx m;
    idx n;
    idx ld;
    T* data;
    static constexpr Layout layout = L;
    using scalar_type = T;

    
    Matrix(T* data, idx m, idx n, idx ld) : data(data), m(m), n(n), ld(ld) {}

    constexpr inline T& operator() const (idx row, idx col) {
       return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((*this)(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((*this)(i,j))) return true;
            }
        }
        return false;
    }

    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }

};

template<typename T, typename idx, typename T_scal , Layout L = ColMajor>
class MX_Matrix {
    
    idx m;
    idx n;
    idx ld;
    idx r;  //numbver  of floats that share an exp
    T* data;
    T_scal* shared_exps;
    static constexpr Layout layout = L;
    using scalar_type = T;
    using shared_exp_type = T_scal;

    
    MX_Matrix(T* data, T_scal* shared_exp, idx m, idx n, idx ld, idx r) : data(data), shared_exp(shared_exp), m(m), n(n), ld(ld), r(r) {}

    constexpr inline idx get_idx(idx row, idx col) const {
        if constexpr (L == ColMajor) return col*ld + row;
        else return row*ld + col;
    }

    constexpr inline T& operator() const (idx row, idx col) {
        return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    } 

    constexpr inline T_scal get_exp(idx row, idx col) const {
        if constexpr (L == Layout::ColMajor) {
            return T_scal[(col*ld + row)/r];
        } else {
            return T_scal[(row*ld + col)/r];
        }
    }

    template<typename T>
    constexpr inline void set_exp(idx row, idx col, T value) const {

        if constexpr (L == Layout::ColMajor) {
            T_scal[(col*ld + row)/r] = std::is_integral_v<shared_exp_type> ? static_cast<shared_exp_type>(log2(value)) : static_cast<shared_exp_type>(value);
        } else {
            T_scal[(row*ld + col)/r] = std::is_integral_v<shared_exp_type> ? static_cast<shared_exp_type>(log2(value)) : static_cast<shared_exp_type>(value);
        }
    }

    template<typename V>
    constexpr inline V scaled_val(idx row, idx col) const {
        return operator()(row, col) * get_exp(row, col);
    }


    constexpr inline T& operator() const (idx row, idx col) {
        return L == ColMajor ? data[col*ld + row] : data[row*ld + col];
    }

    bool isNaN() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isnan((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }

    bool isInf() const {
        for(int row = 0; row < m; row++) {
            for(int col = 0; col < n; col++) {
                if(isinf((this)->scaled_val(i,j))) return true;
            }
        }
        return false;
    }


    constexpr inline idx rows() const {
        return this->m;
    }

    constexpr inline idx cols() const {
        return this->n;
    }



};

template<typename MatrixA, typename MatrixA_t, int n_block_size, int m_block_size>
void transpose(MatrixA& A, MatrixA_t& At) {

    At.m = A.n;
    At.n = A.m;
    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;
    if constexpr (At::layout != A::layout) {
        //just copy over the data with memcpy
        std::memcpy(At.data, A.data, A.m*A.n*sizeof(typename MatrixA::value_type));
    } else {
        // block transpose
        for(int b_row = 0; b_row < n_blocks; b_row++) {
            for(int b_col = 0; b_col < m_blocks; b_col++) {
                for(int row = b_row*n_block_size; row < std::min(A.m, row + n_block_size); row++) {
                    for(int col = b_col*m_block_size; col < std::min(A.n, col + m_block_size); col++) {
                            At.data[At.get_idx(row, col)] = A.data[A.get_idx(col, row)];
                    }
                }
            }
        }
    }
    
}


//main edge case here is when A.r != At.r
template<typename MX_MatrixA, typename MX_MatrixAt, int n_block_size, int m_block_size>
void transpose(MX_MatrixA& A, MX_MatrixAt& At)
{
    At.m = A.n;
    At.n = A.m;

    using A_type = MX_MatrixA::scalar_type;

    int n_blocks = (A.n + n_block_size - 1) / n_block_size;
    int m_blocks = (A.m + m_block_size - 1) / m_block_size;

    if constexpr (A::layout != At::layout) {
        std::memcpy(At.data, A.data, A.m*A.n*sizeof(typename MatrixA::value_type));
        if constexpr (A.r = At.r) {
            std::memcpy(A.shared_exp, At.shared_exp, A.m*A.n*sizeof(typename MatrixA::value_type)/A.r);

        } else {
            //traverse At and set exponents
            if constexpr (At::layout == Layout::ColMajor) {
                for(int i = 0; i < At.n; i++) {
                    //parallelize with OpenMP
                    for(int j_block = 0; j_block < At.m/r; j++) {
                        //find max
                        auto blk_max = static_cast<A_type>(0.0);
                        for(int j = j_block*r; j < std::min(At.m, j + r); j++) {
                            blk_max = std::max(blk_max, At(i,j));
                        }
                        //set blk_max to exp array -

                        

                    }
                }
            } else {
                for(int i = 0; i < At.m; i++) {
                    for(int j = 0; j < At.n; j++) {

                    }
                }
            }
        }
    } else {

    }


}



//helper that returns if the format is MX by checking if the type has a shared_exps member
template<typename T>
struct is_MX_format {
    static constexpr bool value = false;
};
template<typename T, typename idx, typename T_scal, Layout L, MX_Layout MX_L>
struct is_MX_format<MX_Matrix<T, idx, T_scal, L, MX_L>> {
    static constexpr bool value = true;
};
template<typename T>
struct is_MX_format<Matrix<T, idx>> {
    static constexpr bool value = false;
};









} //namespace Lo_gemm