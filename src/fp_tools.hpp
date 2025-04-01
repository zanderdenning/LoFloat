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

enum Signedness : uint8_t {
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
};

enum SubNormal_Support : uint8_t {
    Has_SubNormal_Support = 0,
    No_SubNormal_Support = 1
};

// Concept: "InfChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept InfChecker = requires(T t, uint32_t bits) {
    // Must be callable with a uint32_t, returning something convertible to bool
    { t(bits) } -> std::convertible_to<bool>;

    // Must have an infBitPattern() that returns something convertible to uint32_t
    { t.infBitPattern() } -> std::convertible_to<uint32_t>;

    // Must have a minNegInf() method returning something convertible to uint32_t
    { t.minNegInf() } -> std::convertible_to<uint32_t>;

    // Must have a minPosInf() method returning something convertible to uint32_t
    { t.minPosInf() } -> std::convertible_to<uint32_t>;
};

// Concept: "NaNChecker" means a type T has operator()(uint64_t) -> bool
template <typename T>
concept NaNChecker = requires(T t, uint32_t bits) {
    { t(bits) } -> std::convertible_to<bool>;
    { t.qNanBitPattern() } -> std::convertible_to<uint32_t>;
    { t.sNanBitPattern() } -> std::convertible_to<uint32_t>;

};



template<InfChecker IsInfFunctor, NaNChecker IsNaNFunctor>
struct FloatingPointParams
{
    int bitwidth;
    int mantissa_bits;
    int bias;
    Rounding_Mode rounding_mode;
    Inf_Behaviors OV_behavior;
    NaN_Behaviors NA_behavior;
    Signedness is_signed;
    SubNormal_Support SN_support;
    IsInfFunctor IsInf;
    IsNaNFunctor IsNaN;

    constexpr FloatingPointParams(
        int bw,
        int mb,
        int b,
        Rounding_Mode rm,
        Inf_Behaviors ovb,
        NaN_Behaviors nab,
        Signedness is_signed,
        SubNormal_Support SN_sup,
        IsInfFunctor IsInf,
        IsNaNFunctor IsNaN
    )
      : bitwidth(bw)
      , mantissa_bits(mb)
      , bias(b)
      , rounding_mode(rm)
      , OV_behavior(ovb)
      , NA_behavior(nab)
      , is_signed(is_signed)
      , SN_support(SN_sup)
      , IsInf(IsInf)
      , IsNaN(IsNaN)
    {}
};

}