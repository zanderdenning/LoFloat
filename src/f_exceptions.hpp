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
#include <limits>
#include <mutex>


//need some kind of functors that tell us how to deal with traps. Also need some global flags that are visible to users

namespace lo_float {

    namespace lo_float_internal {

        //need to set DivisionByZero in the parent class along with arith ops. Might need to set it in lo_float_sci as well
        //Set Overflow, underflow in the convert function. I count going from negative number to 0 in the case of unsigned floats as an underflow
        //Set InvalidOperation according to 754 doc and also provide some way to allow users to define operations/wrappers that let users define their own invalid ops
        //Set inexact in the conversion function when round the mantissa. If the rounded bits are the exact same as the original bits, then we don't set inexact


        //all flags are sticky
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

        // trap handler alias 
        using TrapHandler = void(*)(LF_exception_flags) noexcept;

        





        struct Environment {
            #ifdef MLT 
            static std::atomic<uint8_t> exception_flags = static_cast<uint8_t>(NoException);
            #else
            uint8_t exception_flags;
            #endif
            uint8_t trapping_flags;    //trapping flags set to 0 

            Environment() : exception_flags(static_cast<uint8_t>(NoException)), trapping_flags(static_cast<uint8_t>(NoException)) {}


        };




        // Function to set exception flags
        void set_exception_flags(LF_exception_flags flags) {
            exception_flags |=  static_cast<uint8_t>(flags);
        }

        void clear_exception_flags(LF_exception_flags flags) {
            exception_flags &= ~static_cast<uint8_t>(flags);
        }
        
        void reset_exception_flags() {
            exception_flags = static_cast<uint8_t>(NoException);
        }

        

        // default implementation: print + abort  ───────────────────────────────────
        [[noreturn]] inline void default_trap(LF_exception_flags f) noexcept
        {
            std::fputs("lo_float trap: ", stderr);
            std::bitset<8> bits(static_cast<std::uint8_t>(f));
            std::fprintf(stderr, "flags=%s | thread=%zu\n",
                        bits.to_string().c_str(),
                        static_cast<size_t>(std::hash<std::thread::id>{}(
                                std::this_thread::get_id())));
            std::abort();
        }

        
        inline std::atomic<TrapHandler> current_trap_handler{ default_trap };

        // API                                                                       ─
        template<Trap H>
        void set_trap_handler(H&& h) noexcept
        {
            current_trap_handler.store(+h, std::memory_order_release);
        }

        [[nodiscard]] inline TrapHandler get_trap_handler() noexcept
        {
            return current_trap_handler.load(std::memory_order_acquire);
        }

        // invoke helper: never returns                                              ─
        [[noreturn]]
        inline void raise_trap(LF_exception_flags f) noexcept
        {
            get_trap_handler()(f);
            std::terminate();               // belt & braces if user returns
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
