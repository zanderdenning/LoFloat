///@author Sudhanva Kulkarni
///header file containing a bunch of useful enum and struct definitions
#include <cstdint>

#include <concepts>
#include <cstdint>



namespace lo_float {    


enum Rounding_Mode : uint8_t {
    RoundToNearestEven = 0,
    RoundTowardsZero = 1,
    RoundAwayFromZero = 2,
    StochasticRounding = 3,
    RoundToNearestOdd = 4
};

enum Signed_Type : uint8_t {
    Signed = 0,
    Unsigned = 1
};

enum Inf_Behaviors : uint8_t {
    HasNonTrappingInf = 0,
    Saturating = 1,
    Trapping = 2
};

enum NaN_Behaviors : uint8_t {
    HasQuietNaN = 0,
    NoNaN = 1,
    SignalingNaN = 2
}

// Concept: "InfChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept InfChecker = requires(T t, uint32_t bits) {
    { t(bits) } -> std::convertible_to<bool>;  
    // or -> std::same_as<bool>; if you require exactly bool
};

// Concept: "NaNChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept NaNChecker = requires(T t, uint32_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
};

    //struct to define floating point params. You can specify bitwidth, #mantissa bits, #exp bits, signed or unsigned, 
    template<InfChecker IsInfFunctor, NaNChecker IsNaNFunctor>
    typedef struct {
        int bitwidth;
        int mantissa_bits;
        Signed_Type signed;
        int bias;
        Rounding_Mode rounding_mode;
        Inf_Behaviors OV_behavior;
        NaN_Behaviors NA_behavior;
        IsInfFunctor IsInf;
        IsNaNFunctor IsNaN;
    } FloatingPointParams;
}