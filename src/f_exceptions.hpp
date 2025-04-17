///@author Sudhanva Kulkarni, UC Berkeley
/*
    * @file
    * @brief This file contains the definition of the FloatingPointException class, which is used to handle exceptions related to floating-point operations.
    * 
    * @note This file is part of the lo_float library, which provides a set of tools for working with low to medium floating-point numbers in C++.
*/

#include <cstdint>
#include <concepts>
#include <iostream>


//need some kind of functors that tell us how to deal with traps. Also need some global flags that are visible to users

namespace lo_float {

    namespace lo_float_internal {


        enum LF_exception_flags : uint8_t {
            // No exception
            NoException = 0,

            // Floating-point exceptions
            DivisionByZero = 1 << 0,
            Overflow = 1 << 1,
            Underflow = 1 << 2,
            InvalidOperation = 1 << 3,
            Inexact = 1 << 4,

            // All exceptions
            AllExceptions = DivisionByZero | Overflow | Underflow | InvalidOperation | Inexact
        };

        static LF_exception_flags exception_flags = NoException;

        // Function to set exception flags
        void set_exception_flags(LF_exception_flags flags) {
            exception_flags = (LF_exception_flags) (exception_flags | flags);
        }

        void clear_exception_flags(LF_exception_flags flags) {
            exception_flags = (LF_exception_flags) (exception_flags & ~flags);
        }
        
        void reset_exception_flags() {
            exception_flags = NoException;
        }

        void print_exception_flags() {
            if (exception_flags == NoException) {
                std::cout << "No exceptions\n";
            } else {
                std::cout << "Exceptions: ";
                if (exception_flags & DivisionByZero) {
                    std::cout << "DivisionByZero ";
                }
                if (exception_flags & Overflow) {
                    std::cout << "Overflow ";
                }
                if (exception_flags & Underflow) {
                    std::cout << "Underflow ";
                }
                if (exception_flags & InvalidOperation) {
                    std::cout << "InvalidOperation ";
                }
                if (exception_flags & Inexact) {
                    std::cout << "Inexact ";
                }
                std::cout << "\n";
            }
        }

       


    } // namespace lo_float_internal
}
